import argparse
import json
import operator
import os
import sys, traceback
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, CLIPModel
from openslide import OpenSlide
import pandas as pd
import shutil

def get_model_processor(args):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model_path).to(device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    return device, model, processor

def get_feature(device, model, processor, file):
    feature_array = None
    try:
        image = Image.open(file)
        inputs = processor(images=image, return_tensors="pt").to(device)

        image_features = model.get_image_features(**inputs)
        feature_array = image_features.cpu().detach().numpy()[0].tolist()

    except Exception:
        print('Unable to process:', file)
        traceback.print_exc(file=sys.stdout)


    return feature_array

def extract_slides(args):

    #remove old save directory
    if os.path.exists(args.image_output_path):
        shutil.rmtree(args.image_output_path)
    os.mkdir(args.image_output_path)
    print('Creating', args.image_output_path, 'directory')
    os.mkdir(os.path.join(args.image_output_path,'0'))
    os.mkdir(os.path.join(args.image_output_path, '1'))

    #labels for predicted slides
    slide_labels = dict()
    with open(args.validation_dataset_path, 'r') as f:
        dataset = json.load(f)

    print('Opening validation dataset', args.validation_dataset_path, 'with', len(dataset['validation']), 'slides.')
    for slide in dataset['validation']:
        slide_labels[slide['image']] = slide['label']

    #preprocessing data (tissue percentage, score, blur, etc.) per slide tile
    slide_data = dict()
    with open(args.preprocess_dataset_path, 'r') as f:
        dataset = json.load(f)

    print('Opening preprocess dataset', args.preprocess_dataset_path, 'with', len(dataset), 'slides.')

    for slide in dataset:
        if slide['source_file'] in slide_labels:
            slide_data[slide['source_file']] = dict()
            for tile in slide['tiles']:
                x = tile['points']['x_s']
                y = tile['points']['y_s']
                if x != 0:
                    x = int(x/32)
                if y != 0:
                    y = int(y/32)

                point = str(x) + '-' + str(y)
                slide_data[slide['source_file']][point] = tile

    #tile predictions
    tile_predictions = pd.read_csv(args.tile_predictions_path)

    print('Opening prediction dataset', args.tile_predictions_path, 'with', len(tile_predictions), 'tiles.')

    #merge pred with preprocess data
    tile_map = dict()
    for index, row in tile_predictions.iterrows():
        filename = os.path.basename(row['file'])
        tile_x = row['tile_x']
        tile_y = row['tile_y']
        probability = row['probability']
        label = slide_labels[filename]

        point = str(row['tile_x']) + '-' + str(row['tile_y'])
        if filename in slide_data:
            if point in slide_data[filename]:
                if filename not in tile_map:
                    tile_map[filename] = []
                tile = dict()
                tile['tile_x'] = tile_x
                tile['tile_y'] = tile_y
                tile['probability'] = probability
                if label == 0:
                    tile['probability'] = 1 - probability
                tile['label'] = label
                tile['tp'] = slide_data[filename][point]['tp']
                tile['cf'] = slide_data[filename][point]['cf']
                tile['scf'] = slide_data[filename][point]['svf']
                tile['qf'] = slide_data[filename][point]['qf']
                tile['s'] = slide_data[filename][point]['s']
                tile['br'] = slide_data[filename][point]['br']
                tile['prob_tp'] = ((tile['tp']/100) + probability)/2
                tile_map[filename].append(tile)

            else:
                print('Point should always exists!')
                print(point)
                print(filename)
                for tile_id, tile in slide_data[filename].items():
                    print(tile_id)
                exit(0)

    # extract resulting set
    for filename, tile_list in tile_map.items():
        slide_path = os.path.join(args.slide_path,filename)
        if os.path.isfile(slide_path):
            slide = OpenSlide(slide_path)

            #sort by pred_tp and take top 5
            tile_list = sorted(tile_list, key=operator.itemgetter('prob_tp'), reverse=True)
            tile_list = tile_list[0:5]

            for tile in tile_list:

                region = (tile['tile_x'], tile['tile_y'])
                level = 5
                size = (768, 768)
                im_tile = slide.read_region(region, level, size)

                save_file = os.path.splitext(filename)[0] + '_' + str(tile['tile_x']) + '_' + str(tile['tile_y']) + '.jpg'
                save_file_path = os.path.join(args.image_output_path,str(tile['label']),save_file)
                rgb_im_tile = im_tile.convert('RGB')
                rgb_im_tile.save(save_file_path)
        else:
            print('slide',slide_path,'not found.')

    # Serializing json
    json_object = json.dumps(tile_map, indent=4)

    # Writing to sample.json
    with open(args.report_output_path, "w") as outfile:
        outfile.write(json_object)

def extract_image_files(args):

    feature_extract_results = []

    # get extract list
    canidate_files = list(Path(args.input_path).rglob("*.*"))

    # if there are files in the list process them
    if len(canidate_files) > 0:

        # get models
        device, model, processor = get_model_processor(args)

        for file_path in canidate_files:
            file_path = str(file_path)
            # get features
            feature_array = get_feature(device, model, processor, file_path)
            if feature_array is not None:
                print('extracted features:', file_path)
                feature_data = dict()
                feature_data['file_path'] = file_path
                feature_data['features'] = feature_array
                feature_extract_results.append(feature_data)

    if len(feature_extract_results) > 0:
        # Serializing json
        json_object = json.dumps(feature_extract_results, indent=4)

        # Writing to sample.json
        with open(args.output_path, "w") as outfile:
            outfile.write(json_object)

        #convert to ml format
        df_list = []
        for extract in feature_extract_results:

            file_path = extract['file_path']
            features = [extract['features']]

            print('processing:', file_path)

            new_df = pd.DataFrame(data=features)
            if '1' in file_path:
                new_df['class'] = 1
            else:
                new_df['class'] = 0

            df_list.append(new_df)

        df = pd.concat(df_list)
        df = df.reset_index()
        df = df.drop('index', axis=1)
        df.index.name = 'index'

        df.to_csv(args.ml_output_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CLIP Image Feature Extractor')
    parser.add_argument('--model_path', type=str, default='openai/clip-vit-base-patch32', help='name of project')
    parser.add_argument('--input_path', type=str, default='images', help='name of project')
    parser.add_argument('--output_path', type=str, default='extracts.json', help='name of project')
    parser.add_argument('--ml_output_path', type=str, default='ml_extracts.csv', help='name of project')

    #using openslide
    parser.add_argument('--prediction_threshold', type=float, default=0.8, help='name of project')
    parser.add_argument('--slide_path', type=str, default='slides', help='name of project')
    parser.add_argument('--validation_dataset_path', type=str, default='val_colon_dataset.json', help='name of project')
    parser.add_argument('--preprocess_dataset_path', type=str, default='extract_dump.json', help='name of project')
    parser.add_argument('--tile_predictions_path', type=str, default='tile_predictions_val.csv', help='name of project')
    parser.add_argument('--image_output_path', type=str, default='extracted_images', help='name of project')
    parser.add_argument('--report_output_path', type=str, default='image_report.json', help='name of project')

    parser.add_argument(
        "--slide_extract",
        action="store_true",
        help="run only inference on the validation set, must specify the checkpoint argument",
    )

    # get args
    args = parser.parse_args()

    if args.slide_extract:
        extract_slides(args)
    else:
        extract_image_files(args)
