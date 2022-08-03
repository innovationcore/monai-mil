import json
import pandas as pd
from sklearn.model_selection import train_test_split


def createDatasetJson():
    slide_dataset = dict()
    slide_dataset['training'] = []
    slide_dataset['validation'] = []

    df = pd.read_csv('source_slides.csv')
    label_map = {'M': 2, 'I': 1, 'B': 0}

    train, test = train_test_split(df, test_size=0.2, stratify=df['label'])

    for row_set, row_set_name in [(train, 'training'), (test, 'validation')]:
        for index, row in row_set.iterrows():
            slide = dict()
            slide['image'] = str(row['file']) + '.svs'
            slide['label'] = label_map[row['label']]
            slide_dataset[row_set_name].append(slide)

            if slide['label'] == 1:
                print(slide)

    # Serializing json
    json_object = json.dumps(slide_dataset, indent=4)

    # Writing to sample.json
    with open("dataset_slides.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == '__main__':
    createDatasetJson()
