'''
Korean-Safety-Benchmarks
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
'''

import os
import json
import random
import logging
import argparse

from tqdm import tqdm

from pipeline.hyperclova.hyperclova_completor import HyperCLOVACompletor
from pipeline.utils import KOSBI_PROMPTS as PROMPTS

random.seed(42)

logger = logging.getLogger('context_generation')
logger.setLevel(logging.DEBUG)


def select_from_pool(pool, current_demo, n):
    random.shuffle(pool)
    cnt = 0
    for d in pool:
        if cnt >= n:
            break

        if (d['demographic_category'], d['demographic_group'], d['context']) not in current_demo:
            current_demo.append(
                (d['demographic_category'], d['demographic_group'], d['context'])
            )
            cnt += 1

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # load HyperCLOVACompletor
    api = HyperCLOVACompletor(
        args.api_url,
        args.api_key,
        max_tokens=args.max_tokens,
        repeat_penalty=args.repeat_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # load demonstrations
    context_demo = json.load(open(args.demo_fpath, 'r'))
    logger.info(f"# of demonstrations : {len(context_demo)}")

    # load all list of demographical categories and groups
    demographic_list = json.load(open(args.demographic_list, 'r'))

    # prepare prompts
    prompts = PROMPTS['context']

    generation_results = []
    for category, group_sets in tqdm(demographic_list.items()):
        
        for group_set in tqdm(group_sets):
            groups = [g.strip() for g in group_set.split(',')]

            # prepare demonstrations from same category
            demo_in_category = \
                [x for x in context_demo if x['demographic_category'] == category]
            
            # prepare demonstrations from same group
            demo_in_group = \
                [x for x in context_demo if x['demographic_group'] in groups]

            output_contexts = []
            for _ in range(args.num_generations):
                temperature = args.temperature
                top_p = args.top_p

                # everytime we re-generate, we gradually increase temp and top-p
                temperature_step = (args.max_temperature - args.temperature) / args.retry_tolerance
                top_p_step = (args.max_top_p - args.top_p) / args.retry_tolerance

                retry = 0

                while True:
                    if retry > args.retry_tolerance:
                        logger.warning((
                            f"SKIP : {args.retry_tolerance} retries were generated "
                            f"for this (category, group) pair. Do not generate any more for this "
                            f"- ({category}, {group_set})"
                        ))
                        output_context = ""    # meaning skipping
                        break

                    demo = []

                    # select from same category demo pool
                    select_from_pool(demo_in_category, demo, args.num_in_cat_demos)

                    # select from same group demo pool
                    select_from_pool(demo_in_group, demo, args.num_in_grp_demos)

                    # randomly select (num_total_demos - len(demos)) from entire demo pool
                    select_from_pool(context_demo, demo, args.num_total_demos - len(demo))

                    assert len(demo) == args.num_total_demos

                    input_text = ""

                    # prompt starts with an instruction
                    input_text += prompts['instruction']
                    input_text += "\n"

                    # put demonstrations into prompt
                    random.shuffle(demo)
                    for demo_cat, demo_grp, demo_context in demo:
                        input_text += prompts['context'].format(
                            category=demo_cat,
                            group=demo_grp,
                            context=demo_context,
                        )
                        input_text += "\n"

                        input_text += "###"
                        input_text += "\n"

                    # now, pick a group from groups
                    group = random.sample(groups, 1)[0]

                    input_text += prompts['context'].format(
                        category=category,
                        group=group,
                        context=""
                    )

                    # generate context
                    prompt_len = len(input_text.strip())
                    output_context = api.generate(
                        input_text.strip(),
                        temperature=temperature,
                        top_p=top_p,
                    )[prompt_len:].strip()

                    # check duplication
                    if output_context not in output_contexts:
                        output_contexts.append(output_context)
                        break

                    retry += 1
                    logger.warning(
                        f"RE-GEN : {retry} out of {args.retry_tolerance} / {output_context}"
                    )

                    # temperature and top-p increased to avoid re-generation !!
                    temperature += temperature_step
                    top_p += top_p_step

                if output_context:
                    generation_results.append({
                        "demographic_category": category,
                        "demographic_group_set": group_set, 
                        "demographic_group": group,
                        "context": output_context,
                    })

        # save if all samples are generated per group 
        with open(os.path.join(args.output_dir, args.output_fname), 'w', encoding="utf-8") as fp:
            json.dump(generation_results, fp, indent=4, ensure_ascii=False)

    with open(os.path.join(args.output_dir, args.output_fname), 'w', encoding="utf-8") as fp:
        json.dump(generation_results, fp, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # argument

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url",
                        type=str)
    parser.add_argument("--api_key",
                        type=str)
    parser.add_argument("--demo_fpath",
                        type=str)
    parser.add_argument("--num_total_demos",
                        type=int,
                        default=10)
    parser.add_argument("--num_in_cat_demos",
                        type=int,
                        default=5)
    parser.add_argument("--num_in_grp_demos",
                        type=int,
                        default=3)
    parser.add_argument("--demographic_list",
                        type=str)
    parser.add_argument("--num_generations",
                        type=int,
                        default=50)
    parser.add_argument("--output_dir",
                        type=str,
                        default="data/generations")
    parser.add_argument("--output_fname",
                        type=str,
                        default="output_contexts.json")
    parser.add_argument("--retry_tolerance",
                        type=int,
                        default=10)

    parser.add_argument("--max_tokens",
                        type=int,
                        default=50)
    parser.add_argument("--repeat_penalty",
                        type=float,
                        default=5.0)
    parser.add_argument("--temperature",
                        type=float,
                        default=0.5)
    parser.add_argument("--max_temperature",
                        type=float,
                        default=1)
    parser.add_argument("--top_p",
                        type=float,
                        default=0.8)
    parser.add_argument("--max_top_p",
                        type=float,
                        default=0.95)
    args = parser.parse_args()

    # logging

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(stream_handler)
    
    main(args)