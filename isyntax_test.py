import argparse
import json
import os
import traceback
from os.path import exists

import openslide
import numpy as np
from enum import Enum
import sys


# def check_slide(args, slide_path):
#    print(slide_path)
#    return True

def check_slide(args, slide_path):
    slide = openslide.open_slide(slide_path)
    tile_size = 224
    random_offset = (0, 0)

    np.random.seed(1)

    sum_pixels = 0
    level_dimensions = slide.level_dimensions[args.level]
    num_tiles_x = level_dimensions[0] // tile_size
    num_tiles_y = level_dimensions[1] // tile_size
    #print(f'{num_tiles_x=} {num_tiles_y=}')
    for j in range(num_tiles_y):
        for i in range(num_tiles_x):
    #        print(f'reading {args.level=} {i=} {j=}')
            image = slide.read_region((i * tile_size, j * tile_size), args.level, (tile_size, tile_size))
            sum_pixels += np.sum(np.array(image))
    #print(f'{sum_pixels=}')
    return sum_pixels

if __name__ == "__main__":
    """Reads the whole slide to test ML case & memory consumption."""

    # create arg parser
    parser = argparse.ArgumentParser(description='read data file and test')

    # general args
    parser.add_argument('--data_root', type=str, help='name of project', default='/slides')
    parser.add_argument('--bad_slide_list', type=str, help='name of project', default='bad_slides.json')
    parser.add_argument('--dataset_slides_src', type=str, help='name of project', default='full_dataset_slides.json')
    parser.add_argument('--dataset_slides_dst', type=str, help='name of project', default='dataset_slides.json')
    parser.add_argument('--level', type=int, default=6)

    # get args
    args = parser.parse_args()

    if (args.data_root is not None) and (args.dataset_slides_src is not None) and (args.dataset_slides_dst is not None):

        new_dataset = dict()

        if exists(args.bad_slide_list):
            f = open(args.bad_slide_list)
            bad_slides = json.load(f)

        f = open(args.dataset_slides_src)
        dataset = json.load(f)
        total = 0
        missing = 0
        added = 0
        broken = 0
        for key, value in dataset.items():
            new_dataset[key] = []
            for slide in value:
                total += 1
                slide_path = os.path.join(args.data_root, slide['image'])
                if exists(slide_path):
                    if slide['image'] in bad_slides:
                        print('Known Bad: ' + slide_path)
                        broken += 1
                    else:
                        print('Checking: ' + slide_path)
                        try:
                            check_size = check_slide(args, slide_path)
                            if check_size > 0:
                                # print('Adding: ' + slide_path + ' size: ' + str(check_size))
                                new_dataset[key].append(slide)
                                added += 1
                            else:
                                print('Not found ' + slide_path)
                                broken += 1
                        except:
                            print('Found Bad: ' + slide_path)
                            traceback.print_exc()
                            broken += 1
                else:
                    print('Missing: ' + slide_path)
                    missing += 1

        json_object = json.dumps(new_dataset, indent=4)

        # Writing to sample.json
        with open(args.dataset_slides_dst, "w") as outfile:
            outfile.write(json_object)

        print('Total slides: ' + str(total))
        print('Added slides: ' + str(added))
        print('Missing slides: ' + str(missing))
        print('Broken slides: ' + str(broken))

