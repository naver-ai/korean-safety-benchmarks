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

import numpy as np

from pipeline.utils import load_filter_models


logger = logging.getLogger('response_demo_augmentation')
logger.setLevel(logging.DEBUG)

def check_all_agree(raw_annotations, query="Q2: Acceptable or Non-acceptable"):
    annotations = raw_annotations[query]
    return len(set([a['acceptable?'] for a in annotations])) == 1

def variability(pipes, dataset, class_id):
    model_inputs = [
        d['question'] + " " + d['response']
        for d in dataset
    ]

    preds = [[[], []] for _ in range(len(model_inputs))]

    for midx, pipe in enumerate(pipes):
        print(f"{midx+1}th model inference...")
        epoch_preds = pipe(model_inputs)
        for i, epoch_pred in enumerate(epoch_preds):
            for j in range(2):
                preds[i][j].append([p['score'] for p in epoch_pred][j])

    vars = [
        np.std(pred[class_id])
        for pred in preds
    ]

    return vars

def main(args):
    # load generated responses
    input_dataset = json.load(open(args.input_fname, 'r'))

    # load previous demo file
    if args.prev_demo_fname is not None:
        prev_demo = json.load(open(args.prev_demo_fname, 'r'))
    else:
        prev_demo = []

    # start from previous demo file
    output_demo = deepcopy(prev_demo)

    # augment data only all annotators agree
    output_demo += [
        d for d in input_dataset
        if check_all_agree(d['raw_annotations'])
    ]

    # load filtering model
    pipes = load_filter_models(
        args.filter_model_type,
        args.filter_model_ckpt_dir,
        args.filter_model_train_epochs)

    output_demo_amb = []

    # select 'ambiguous' sample in demonstrations of each response class
    for class_id, demos in enumerate([
        [d for d in output_demo if not d['acceptable?']],
        [d for d in output_demo if d['acceptable?']],
    ]):
        # calculate max variability
        logger.info(f"prediction starts")
        vars = variability(pipes, demos, class_id)

        # prepare demo for each question category
        for question_category in ['contentious', 'ethical', 'predictive', 'etc']:
            _demos = [
                (d,v) for d,v in zip(demos, vars)
                if d['question_category'] == question_category
            ]
            
            # sort and index
            _demos = sorted(_demos, key=lambda d: d[1], reverse=True)
            ratio = int(len(_demos)*args.most_ambiguous_ratio)
            output_demo_amb += [d[0] for d in _demos[:ratio]]

    logger.info(f"before : {len(prev_demo)} / after : {len(output_demo_amb)}")

    with open(os.path.join(args.output_dir, args.output_demo_fname), 'w', encoding="utf-8") as fp:
        json.dump(output_demo_amb, fp, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # argument

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fname",
                        type=str)
    parser.add_argument("--prev_demo_fname",
                        type=str)
    parser.add_argument("--filter_model_type",
                        type=str,
                        default="beomi/KcELECTRA-base-v2022")
    parser.add_argument("--filter_model_ckpt_dir",
                        type=str,
                        default="data/models")
    parser.add_argument("--filter_model_train_epochs",
                        type=int,
                        default=5)
    parser.add_argument("--output_dir",
                        type=str,
                        default="pipeline/demo")
    parser.add_argument("--output_demo_fname",
                        type=str,
                        default="square_response_iter1.json")
    parser.add_argument("--most_ambiguous_ratio",
                        type=float,
                        default=0.25)
    args = parser.parse_args()

    # logging

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(stream_handler)
    
    main(args)