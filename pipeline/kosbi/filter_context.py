'''
Korean-Safety-Benchmarks
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
'''

import os
import json
import logging
import argparse

from tqdm import tqdm

from pipeline.utils import load_filter_model

logger = logging.getLogger('context_filtering')
logger.setLevel(logging.DEBUG)


def main(args):
    # load filtering model
    if args.filter_model_ckpt is not None:
        pipe, tok = load_filter_model(args.filter_model_type, args.filter_model_ckpt, return_tokenizer=True)

    # load generated contexts
    input_dataset = json.load(open(args.input_fname, 'r'))
    
    output_dataset = []
    
    total_cnt = 0
    filtered_cnt = 0
    for data in tqdm(input_dataset, total=len(input_dataset)):
            if args.filter_model_ckpt is not None:
                model_input = data['demographic_group'] \
                    + " " \
                    + tok.sep_token \
                    + " " \
                    + data['context']

                if list(reversed(sorted(pipe(model_input)[0], key=lambda x: x["score"])))[0]['label'] \
                    == "LABEL_1":
                    # means that the context contains the target group
                    output_dataset.append({
                        'demographic_category': data['demographic_category'],
                        'demographic_group': data['demographic_group'],
                        'context': data['context'],
                        'generation_info': {
                            'demographic_group_set': data['demographic_group_set']
                        }
                    })
                else:
                    filtered_cnt += 1
            else:
                output_dataset.append({
                    'demographic_category': data['demographic_category'],
                    'demographic_group': data['demographic_group'],
                    'context': data['context'],
                    'generation_info': {
                        'demographic_group_set': data['demographic_group_set']
                    }
                })
            total_cnt +=1

    logger.info(f"{filtered_cnt}/{total_cnt} are filtered")

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
    parser.add_argument("--filter_model_ckpt",
                        type=str)
    parser.add_argument("--output_dir",
                        type=str,
                        default="data/generations")
    parser.add_argument("--output_fname",
                        type=str,
                        default="filtered_contexts.json")
    args = parser.parse_args()

    # logging

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(stream_handler)
    
    main(args)