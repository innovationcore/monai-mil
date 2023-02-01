# This script does the following:
# 1. Queries the data.ai.uky.edu mongodb for pending job requests.
# 2. Starts MIL training for each job request, with proper sequencing (currently 1 job at a time)
# 3. Allows to stop training for a running job
# 4. Report per-epoch metrics and generally update the job status.

import argparse
import os
import torch
import torch.multiprocessing as mp
import time

from pymongo import MongoClient
from clearml import Logger
from clearml import Task
import numpy as np
from bson.json_util import dumps
import json
from sklearn.model_selection import train_test_split

# Openslide on windows requires a dll. This must be at top level, because monai checks for openslide at top level.
# TODO(avirodov): some args parsing at top level just to get the dll directory?
openslide_dll_path = r'C:/projects/software/openslide-win64-20171122/bin'
if openslide_dll_path is not None and hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(openslide_dll_path):
        import openslide
else:
    import openslide

from MIL import main_worker, add_args


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
    parser.add_argument('--openslide_dll_path', type=str, default=None)
    add_args(parser)
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
    xs = [
        'TCGA-97-7547-01A-01-TS1.85e09741-a86c-46b2-b73d-d04f54df0019.svs',
        'TCGA-75-7025-01A-01-TS1.90ebf9d8-a329-4ae0-b62f-8f79b0b9d800.svs',
        'TCGA-78-7152-01A-01-BS1.17987afb-41a5-44ff-94e2-a282163a10be.svs',
        'TCGA-55-8204-01Z-00-DX1.30ba69f3-53f1-41cc-826c-20dce3cfe86b.svs',
        'TCGA-55-8511-11A-01-TS1.c28c0917-114a-48bb-ae2b-3bb5c057660a.svs',
        'TCGA-55-7724-11A-01-TS1.86e82089-33ea-43ab-8b9c-be619924fbe4.svs',
        'TCGA-75-6205-01A-01-TS1.30141d32-1c26-41d9-8b2e-590b32cb59b6.svs',
        'TCGA-44-8119-11A-01-TS1.d7257647-1870-456f-869c-a89806caba2a.svs',
        'TCGA-75-7031-01A-01-TS1.ff890643-cd25-4528-ab56-1d866053de60.svs',
        'TCGA-91-6828-01A-01-TS1.9582619f-ecd4-4890-8af1-14e5eb0f536e.svs',
        'TCGA-78-7220-01A-01-TS1.322a58dc-7371-4745-90f7-395370a8fd53.svs',
        'TCGA-55-8505-01Z-00-DX1.D364C30D-BFB8-486B-A0D3-948FF8E90C3E.svs',
        'TCGA-78-7540-01A-01-BS1.79380837-059e-4a6c-b052-7adafbca468c.svs',
        'TCGA-78-7163-01A-01-TS1.4b63c412-8fd3-4353-abcd-b9b866b36f87.svs',
        'TCGA-55-6642-01A-01-TS1.7451219e-0c43-4268-8c42-15f186f66fa3.svs',
        'TCGA-55-8301-11A-01-TS1.cbe043d9-a89d-4c4c-a6fa-ca5268334b83.svs',
        'TCGA-55-7281-11A-01-TS1.199c495b-020e-42c8-9197-880d6069dab4.svs',
        'TCGA-78-7540-01A-01-TS1.aa2d97a4-60a1-496f-b7c3-58b887d174ce.svs',
        'TCGA-55-8514-11A-01-TS1.0f2a46b7-61e1-4991-948b-5407896bacd7.svs',
        'TCGA-78-7535-01A-01-BS1.bece83ec-80c6-4163-b489-2495f624cd06.svs',
    ]
    ys = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    print(f'{len(xs)=}')
    print(f'{len(ys)=}')

    val_percentage = int(job_document["task"]["parameters"]["val_percentage"])
    test_percentage = int(job_document["task"]["parameters"]["test_percentage"])
    epochs = int(job_document["task"]["parameters"]["epochs"])
    args.epochs = epochs

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
    args.dataset_json = mil_dataset_filename

    # Start MIL code
    jobs_collection.update_one({"_id": job_document["_id"]},
                               {"$set": {f"status.{int(time.time())}": 'Job started'}})
    print("Argument values:")
    for k, v in vars(args).items():
        print(k, "=>", v)
    print("-----------------")
    def epoch_end_callback(epoch, metrics):
        result_dict = {
            "epoch": epoch,
        }
        result_dict.update(metrics)
        jobs_collection.update_one({"_id": job_document["_id"]}, {"$push": {"results": result_dict}})

    args.epoch_end_callback = epoch_end_callback

    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        args.optim_lr = ngpus_per_node * args.optim_lr / 2  # heuristic to scale up learning rate in multigpu setup
        args.world_size = ngpus_per_node * args.world_size

        print("Multigpu", ngpus_per_node, "rescaled lr", args.optim_lr)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        main_worker(0, args)
    #
    # for epoch in range(10):
    #     train_acc = 1.0 - np.exp(-epoch)
    #     val_acc = 0.9 * (1.0 - np.exp(-epoch))
    #     Logger.current_logger().report_scalar("ACC", "train_acc", iteration=epoch, value=train_acc)
    #     Logger.current_logger().report_scalar("ACC", "val_acc", iteration=epoch, value=val_acc)

    # Update job status to 'complete'.
    jobs_collection.update_one({"_id": job_document["_id"]},
                               {"$set": {f"status.{int(time.time())}": 'Job complete'}})




if __name__ == "__main__":
    main()