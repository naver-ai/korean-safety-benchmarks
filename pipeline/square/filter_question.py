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

logger = logging.getLogger('question_filtering')
logger.setLevel(logging.DEBUG)


def main(args):
    # load filtering model
    if args.filter_model_ckpt is not None:
        pipe = load_filter_model(args.filter_model_type, args.filter_model_ckpt)

    # load generated questions
    input_dataset = json.load(open(args.input_fname, 'r'))
    
    output_dataset = []
    
    total_cnt = 0
    filtered_cnt = 0
    for data in tqdm(input_dataset, total=len(input_dataset)):
        for g in data['generations']:
            if args.filter_model_ckpt is not None:
                if list(reversed(sorted(pipe(g['question'])[0], key=lambda x: x["score"])))[0]['label'] \
                    == "LABEL_1":
                    # subjective
                    output_dataset.append({
                        'question': g['question'],
                        'generation_info': {
                            'topic_source': data['topic_source'],
                            'topic': data['topic'],
                            'keywords': g['keywords'],
                            'standard': g['standard'],
                        }
                    })
                else:
                    filtered_cnt += 1
            else:
                output_dataset.append({
                    'question': g['question'],
                    'generation_info': {
                        'topic_source': data['topic_source'],
                        'topic': data['topic'],
                        'keywords': g['keywords'],
                        'standard': g['standard'],
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
                        default="filtered_questions.json")
    args = parser.parse_args()

    # logging

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(stream_handler)
    
    main(args)