import json
import pandas as pd
from sklearn.model_selection import train_test_split

def createDatasetJson():
    labels = []
    slide_dataset = dict()
    slide_dataset['training'] = []
    slide_dataset['validation'] = []

    df = pd.read_csv('source_slides.csv')

    train, test = train_test_split(df, test_size=0.25, stratify=df['label'])

    for index, row in train.iterrows():
        if row['label'] not in labels:
            labels.append(row['label'])

        slide = dict()
        slide['image'] = str(row['file']) + '.svs'
        slide['label'] = labels.index(row['label'])
        slide_dataset['training'].append(slide)

        if slide['label'] == 2:
            print(slide)

    for index, row in test.iterrows():
        if row['label'] not in labels:
            labels.append(row['label'])

        slide = dict()
        slide['image'] = str(row['file']) + '.svs'
        slide['label'] = labels.index(row['label'])
        slide_dataset['validation'].append(slide)

        if slide['label'] == 2:
            print(slide)

    # Serializing json
    json_object = json.dumps(slide_dataset, indent=4)

    # Writing to sample.json
    with open("dataset_slides.json", "w") as outfile:
        outfile.write(json_object)

if __name__ == '__main__':

    createDatasetJson()

    '''
    slides = ['829812a8-2215-11ec-be26-0242ac110002.tiff']

    for slide in slides:
        slideMetadata(slide)
    '''

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
