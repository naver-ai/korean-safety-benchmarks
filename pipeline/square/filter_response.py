'''
Korean-Safety-Benchmarks
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
'''

import os
import json
import logging
import argparse

import numpy as np

from pipeline.utils import load_filter_models


logger = logging.getLogger('response_filtering')
logger.setLevel(logging.DEBUG)


def est_max_var(pipes, dataset):
    model_inputs = [
        d['question'] + " " + d['response']
        for d in dataset
    ]

    preds = [[[], []] for _ in range(len(model_inputs))]

    for midx, pipe in enumerate(pipes):
        logger.info(f"{midx+1}th pipe...")
        epoch_preds = pipe(model_inputs)
        for i, epoch_pred in enumerate(epoch_preds):
            for j in range(2):
                preds[i][j].append([p['score'] for p in epoch_pred][j])
    
    e_max_var = [
        np.max([np.std(pred[l]) for l in range(2)])
        for pred in preds
    ]

    return e_max_var

def main(args):
    n = args.num_generations_per_question
    
    # load filtering model
    if args.filter_model_ckpt_dir is not None:
        pipes = load_filter_models(
            args.filter_model_type,
            args.filter_model_ckpt_dir,
            args.filter_model_train_epochs)

    # load generated responses
    input_dataset = json.load(open(args.input_fname, 'r'))
    assert all([len(d['generations']) == n for d in input_dataset])

    output_dataset = []
    remaining_dataset = []
    
    total_cnt = 0

    # select 'ambiguous' sample in each response class
    for resp_class in ['response_acceptable', 'response_nonacceptable']:
        response_dataset = [
            {
                'question': d['question'],
                'response': g[resp_class],
                'question_category': d['question_category'],
                'generation_info': {
                    'topic_source': d['topic_source'],
                    'topic': d['topic'],
                }
            }
            for d in input_dataset
            for g in d['generations']
        ]
        total_cnt += len(response_dataset)

        if args.filter_model_ckpt_dir is not None:
            # calculate estimated max variability
            logger.info(f"prediction starts ({resp_class})")
            e_max_var = est_max_var(pipes, response_dataset)

            # select the most 'ambiguous' data among every n generations
            for offset in range(0, len(e_max_var), n):
                argmax = np.argmax(e_max_var[offset:offset+n])
                output_dataset.append(response_dataset[offset:offset+n][argmax])

                # we will sample extra confusing samples
                remaining_dataset += [
                    (d, e_max_var[offset:offset+n][i])
                    for i, d in enumerate(response_dataset[offset:offset+n])
                    if i != argmax
                ]
        else:
            output_dataset += response_dataset

    # if you augment extra confusing samples
    if args.filter_model_ckpt_dir is not None \
        and args.num_extra_ambiguous_samples is not None:
        remaining_dataset = sorted(
            remaining_dataset,
            key=lambda d:d[1],
            reverse=True)
        remaining_dataset = [d[0] for d in remaining_dataset]
        
        output_dataset += remaining_dataset[:args.num_extra_ambiguous_samples]

    logger.info(f"{total_cnt-len(output_dataset)}/{total_cnt} are filtered")

    with open(os.path.join(args.output_dir, args.output_fname), 'w', encoding="utf-8") as fp:
        json.dump(output_dataset, fp, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # argument

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fname",
                        type=str)
    parser.add_argument("--filter_model_type",
                        type=str,
                        default="beomi/KcELECTRA-base-v2022")
    parser.add_argument("--filter_model_ckpt_dir",
                        type=str)
    parser.add_argument("--filter_model_train_epochs",
                        type=int,
                        default=5)
    parser.add_argument("--output_dir",
                        type=str,
                        default="data/generations")
    parser.add_argument("--output_fname",
                        type=str,
                        default="filtered_responses.json")
    parser.add_argument("--num_generations_per_question",
                        type=int,
                        default=3)
    parser.add_argument("--num_extra_ambiguous_samples",
                        type=int)
    args = parser.parse_args()

    # logging

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(stream_handler)
    
    main(args)