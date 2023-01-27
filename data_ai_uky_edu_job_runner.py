# This script does the following:
# 1. Queries the data.ai.uky.edu mongodb for pending job requests.
# 2. Starts MIL training for each job request, with proper sequencing (currently 1 job at a time)
# 3. Allows to stop training for a running job
# 4. Report per-epoch metrics and generally update the job status.

import argparse
import os

from pymongo import MongoClient
from clearml import Logger
from clearml import Task
import numpy as np
from bson.json_util import dumps
import json
from sklearn.model_selection import train_test_split


def find_job_created(jobs_collection):
    for document in jobs_collection.find():
        # print(f'{document=}')
        status_dict = document['status']
        print(f'{status_dict=}')
        # From python 3.7, dictionary order is guaranteed, but I don't want to do a python version check, and sorting
        # a few values should be quick. Also, I'm not sure mongodb/json preserve the order or use python3.7 dicts.
        status_list = sorted(status_dict.items())
        print(f'{status_list=}')
        if status_list[-1][1] == 'Job created':
            return document

    return None

def xs_ys_to_mil_image_label(xs, ys):
    return [{"image": x, "label": y} for x, y in zip(xs, ys)]


def main():
    parser = argparse.ArgumentParser(description='Process jobs from data.ai.uky.edu')
    parser.add_argument('--mongodb_uri', type=str, required=True)
    parser.add_argument('--database', type=str, default="slide_dataset_site_dev")
    args = parser.parse_args()
    print(f'{args=}')
    client = MongoClient(args.mongodb_uri)
    print(f'{client=}')

    if args.database not in client.list_database_names():
        raise LookupError(f'db {args.database=} not found!')
    db = client[args.database]
    print(f'{db=}')

    jobs_collection = db.get_collection('jobs')
    print(f'{jobs_collection=}')

    # Find all jobs with status 'created'. Simple scheduling - pick up the first.
    # TODO(avirodov): We should instead change the format of the 'status' field to make it searchable. E.g.
    #   https://www.mongodb.com/community/forums/t/query-nested-objects-in-a-document-with-unknown-key/14511
    # TODO(avirodov): When picking up jobs, use 'findAndModify()' to ensure atomicity and allow multiple schedulers to
    #   share a db.
    # TODO(avirodov): Handle 'Ctrl+C'.
    # TODO(avirodov): Handle jobs status changed to 'stop' or similar.
    job_document = find_job_created(jobs_collection)
    # print(f'{job_document=}')
    if job_document is None:
        print(f'No new jobs')
        return
    # find_one will return a dict, which is useful for copying it into clearml.
    job_document = jobs_collection.find_one({"_id": job_document["_id"]})
    # print(f'dict {job_document=}')

    # Update job status to running.
    # TODO(avirodov): implement, for now disabled because I need to experiment with clearml.
    # task = Task.init(project_name=args.project_name, task_name=args.task_name)

    # Generate a clearml task and freeze the train/test split as part of it.
    # TODO(avirodov): later allow re-running data.ai.uky.edu jobs, if needed. Then we can use the same split.
    project_name = 'data.ai.uky.edu jobs'
    task_name = f'{job_document["name"]}, {job_document["data"]["name"]}, {job_document["task"]["type"]}'
    print(f'{project_name=} {task_name=}')
    task = Task.init(project_name=project_name, task_name=task_name)
    # The conversion to and from json is to get rid of MongoDB ObjectID objects that ClearML doesn't handle.
    job_document["clearml"] = {
        "project_id": task.project,
        "task_id": task.id,
    }
    jobs_collection.update_one({"_id": job_document["_id"]},
                               {"$set": {"clearml": job_document["clearml"], "results": []}})
    task.upload_artifact('job', json.loads(dumps(job_document)))

    # Extract the slide list per dataset target. Cohorts are 'label' -> {... slides: [....] }, reformat as
    # { slide -> label }
    xs, ys = [], []
    cohorts = job_document["data"]["cohorts"]
    for label, cohort in cohorts.items():
        print(f'{label=}')
        cases = cohort["data"]["cases"]
        slides = [slide for case in cases for slide in case["SLIDES"]]
        # print(f'{slides=}')
        xs.extend(slides)
        ys.extend([label] * len(slides))
    print(f'{len(xs)=}')
    print(f'{len(ys)=}')

    val_percentage = int(job_document["task"]["parameters"]["val_percentage"])
    test_percentage = int(job_document["task"]["parameters"]["test_percentage"])
    print(f'{val_percentage=} {test_percentage=}')
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, stratify=ys,
                                                        test_size=(val_percentage + test_percentage) / 100.0)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, stratify=y_test,
                                                    test_size=val_percentage / (val_percentage + test_percentage))

    # Generate the train-test split and save the generated .json MONAI-MIL dataset.
    mil_datalist_dict = {
        "training": xs_ys_to_mil_image_label(x_train, y_train),
        "validation": xs_ys_to_mil_image_label(x_val, y_val),
        "test": xs_ys_to_mil_image_label(x_test, y_test),
    }
    # print(json.dumps(mil_datalist_dict, indent=2))
    print(f'{len(mil_datalist_dict["training"])=}')
    print(f'{len(mil_datalist_dict["validation"])=}')
    print(f'{len(mil_datalist_dict["test"])=}')
    task.upload_artifact('mil_datalist', mil_datalist_dict)
    os.makedirs('generated_mil_datasets', exist_ok=True)
    mil_dataset_filename = f'generated_mil_datasets/mil_dataset_task_id_{task.id}.json'
    with open(mil_dataset_filename, 'w') as json_file:
        json.dump(mil_datalist_dict, json_file, indent=2)

    # Start MIL code

    for epoch in range(10):
        train_acc = 1.0 - np.exp(-epoch)
        val_acc = 0.9 * (1.0 - np.exp(-epoch))
        Logger.current_logger().report_scalar("ACC", "train_acc", iteration=epoch, value=train_acc)
        Logger.current_logger().report_scalar("ACC", "val_acc", iteration=epoch, value=val_acc)
        result_update = {
            "epoch": epoch,
            "train_acc": train_acc,
            "val_acc": val_acc
        }
        jobs_collection.update_one({"_id": job_document["_id"]}, {"$push": {"results": result_update}})

    # Update job status to 'complete'.




if __name__ == "__main__":
    main()