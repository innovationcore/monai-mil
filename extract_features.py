import argparse
import json
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
    os.mkdir(os.path.join(args.image_output_path,'0'))
    os.mkdir(os.path.join(args.image_output_path, '1'))

    slide_labels = dict()

    with open(args.validation_dataset_path, 'r') as f:
        dataset = json.load(f)

    for slide in dataset['validation']:
        slide_labels[slide['image']] = slide['label']

    tile_predictions = pd.read_csv(args.tile_predictions_path)
    # file, tile_x, tile_y, probability

    tile_map = dict()

    for index, row in tile_predictions.iterrows():
        filename = os.path.basename(row['file'])
        tile_x = row['tile_x']
        tile_y = row['tile_y']
        probability = row['probability']
        label = slide_labels[filename]

        if label == 1:

            if probability >= args.prediction_threshold:
                if filename not in tile_map:
                    tile_map[filename] = []
                tile = dict()
                tile['tile_x'] = tile_x
                tile['tile_y'] = tile_y
                tile['probability'] = probability
                tile['label'] = label
                tile_map[filename].append(tile)

        elif label == 0:

            if probability < (1 - args.prediction_threshold):
                if filename not in tile_map:
                    tile_map[filename] = []
                tile = dict()
                tile['tile_x'] = tile_x
                tile['tile_y'] = tile_y
                tile['probability'] = probability
                tile['label'] = label
                tile_map[filename].append(tile)

    for filename, tile_list in tile_map.items():
        slide_path = os.path.join(args.slide_path,filename)
        if os.path.isfile(slide_path):
            slide = OpenSlide(slide_path)
            for tile in tile_list:

                region = (tile['tile_x'], tile['tile_y'])
                level = 5
                size = (768, 768)
                im_tile = slide.read_region(region, level, size)

                save_file = os.path.splitext(filename)[0] + '_' + str(tile['tile_x']) + '_' + str(tile['tile_y']) + '.jpg'
                save_file_path = os.path.join(args.image_output_path,str(tile['label']),save_file)
                rgb_im_tile = im_tile.convert('RGB')
                rgb_im_tile.save(save_file_path)


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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CLIP Image Feature Extractor')
    parser.add_argument('--model_path', type=str, default='openai/clip-vit-base-patch32', help='name of project')
    parser.add_argument('--input_path', type=str, default='images', help='name of project')
    parser.add_argument('--output_path', type=str, default='extracts.json', help='name of project')

    #using openslide
    parser.add_argument('--prediction_threshold', type=float, default=0.8, help='name of project')
    parser.add_argument('--slide_path', type=str, default='slides', help='name of project')
    parser.add_argument('--validation_dataset_path', type=str, default='val_colon_dataset.json', help='name of project')
    parser.add_argument('--tile_predictions_path', type=str, default='tile_predictions_val.csv', help='name of project')
    parser.add_argument('--image_output_path', type=str, default='extracted_images', help='name of project')
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
