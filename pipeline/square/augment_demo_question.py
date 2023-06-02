'''
Korean-Safety-Benchmarks
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
'''

import os
import json
import logging
import argparse
from copy import deepcopy

from tqdm import tqdm


logger = logging.getLogger('question_demo_augmentation')
logger.setLevel(logging.DEBUG)

def main(args):
    # load generated questions
    input_dataset = json.load(open(args.input_fname, 'r'))

    # load previous demo file
    if args.prev_demo_fname is not None:
        prev_demo = json.load(open(args.prev_demo_fname, 'r'))
    else:
        prev_demo = []

    # start from previous demo file
    output_demo = deepcopy(prev_demo)

    for d in tqdm(input_dataset, total=len(input_dataset)):
        # augment only "sensitive" data for demo
        if not d['sensitive?']:
            continue

        # don't use "etc" category
        if d['category'] == 'etc':
            continue

        # data points what is NOT intended to be "ethical" category but labeled as "ethical"
        # have no "standard" output. So, we remove it for "ethical" category demo
        if d['category'] == "ethical" and not d['generation_info']['standard']:
            continue

        output_demo.append(d)

    logger.info(f"before : {len(prev_demo)} / after : {len(output_demo)}")

    with open(os.path.join(args.output_dir, args.output_demo_fname), 'w', encoding="utf-8") as fp:
        json.dump(output_demo, fp, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # argument

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fname",
                        type=str)
    parser.add_argument("--prev_demo_fname",
                        type=str)
    parser.add_argument("--output_dir",
                        type=str,
                        default="pipeline/demo")
    parser.add_argument("--output_demo_fname",
                        type=str,
                        default="square_question_iter1.json")
    args = parser.parse_args()

    # logging

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(stream_handler)
    
    main(args)