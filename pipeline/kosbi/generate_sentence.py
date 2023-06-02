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

logger = logging.getLogger('sentence_generation')
logger.setLevel(logging.DEBUG)


def select_from_pool(pool, current_demo, n):
    random.shuffle(pool)
    cnt = 0
    for d in pool:
        if cnt >= n:
            break

        if (d['demographic_category'], d['demographic_group'], d['context'], d['sentence']) not in current_demo:
            current_demo.append(
                (d['demographic_category'], d['demographic_group'], d['context'], d['sentence'])
            )
            cnt += 1

def prompting(args, sent_demo, demo_in_context_label, prompts, category, group, context):
    demo = []

    # select from same context label demo pool
    select_from_pool(demo_in_context_label, demo, args.num_in_context_label)

    # randomly select (num_total_demos - len(demos)) from entire demo pool
    select_from_pool(sent_demo, demo, args.num_total_demos - len(demo))

    assert len(demo) == args.num_total_demos

    input_text = ""

    # prompt starts with an instruction
    input_text += prompts['instruction']['safe']
    input_text += "\n"

    # put demonstrations into prompt
    random.shuffle(demo)
    for demo_cat, demo_grp, demo_context, demo_sent in demo:
        input_text += prompts['sentence'].format(
            category=demo_cat,
            group=demo_grp,
            context=demo_context,
            sentence=demo_sent,
        )
        input_text += "\n"

        input_text += "###"
        input_text += "\n"

    # prompt for sentence generation
    input_text += prompts['sentence'].format(
        category=category,
        group=group,
        context=context,
        sentence="",
    )

    return input_text

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
    sent_demo = json.load(open(args.demo_fpath, 'r'))
    safe_demo = [d for d in sent_demo if d['sentence_label'] == "safe"]
    unsafe_demo = [d for d in sent_demo if d['sentence_label'] == "unsafe"]
    
    logger.info(f"# of demonstrations (safe) : {len(safe_demo)}")
    logger.info(f"# of demonstrations (unsafe) : {len(unsafe_demo)}")

    # load contexts
    context_dataset = json.load(open(args.context_fpath, 'r'))

    # we don't generate a next sentence for context NOT including target group
    logger.info(f"before filtering contexts : {len(context_dataset)}")
    context_dataset = [c for c in context_dataset if c['include?']]
    logger.info(f"after filtering contexts : {len(context_dataset)}")

    # prepare prompts
    prompts = PROMPTS['sentence']

    generation_results = []
    for context_data in context_dataset:
        category = context_data['demographic_category']
        group = context_data['demographic_group']
        group_set = context_data['generation_info']['demographic_group_set']
        context = context_data['context']
        context_label = context_data['context_label']
        context_sub_label = context_data['context_sub_label'] \
            if context_data['context_sub_label'] is not None \
            else "none"

        generation_results.append({
            "demographic_category": category,
            "demographic_group_set": group_set,
            "demographic_group": group,
            "context": context,
            "context_label": context_label,
            "context_sub_label": None if context_sub_label == "none" else context_sub_label,
            'generations': [],
        })
        
        # prepare demonstrations from same context label
        safe_demo_in_context_label = \
            [x for x in safe_demo
             if x['context_label'] == context_label and x['context_sub_label'] == context_sub_label]
        unsafe_demo_in_context_label = \
            [x for x in unsafe_demo
             if x['context_label'] == context_label and x['context_sub_label'] == context_sub_label]

        output_safe_list = []
        output_unsafe_list = []
        for _ in range(args.num_generations_per_context):
            temperature = args.temperature
            top_p = args.top_p

            # everytime we re-generate, we gradually increase temp and top-p
            temperature_step = (args.max_temperature - args.temperature) / args.retry_tolerance
            top_p_step = (args.max_top_p - args.top_p) / args.retry_tolerance

            retry = 0

            # generate "safe" sentences
            while True:
                if retry > args.retry_tolerance:
                    logger.warning((
                        f"SKIP : {args.retry_tolerance} retries were generated "
                        f"for this context. Do not generate any more for this context "
                        f"- {context}"
                    ))
                    output_safe = ""    # meaning skipping
                    break
                
                input_text = prompting(
                    args, safe_demo, safe_demo_in_context_label, prompts, category, group, context
                )

                # generate safe sentence
                prompt1_len = len(input_text.strip())
                output_safe = api.generate(
                    input_text.strip(),
                    temperature=temperature,
                    top_p=top_p,
                )[prompt1_len:].strip()

                # check duplication
                if output_safe not in output_safe_list:
                    output_safe_list.append(output_safe)
                    break

                retry += 1
                logger.warning(
                    f"RE-GEN : {retry} out of {args.retry_tolerance} / {output_safe}"
                )

                # temperature and top-p increased to avoid re-generation !!
                temperature += temperature_step
                top_p += top_p_step
            
            # reset
            temperature = args.temperature
            top_p = args.top_p
            retry = 0

            # generate "unsafe" sentences
            while True:
                if retry > args.retry_tolerance:
                    logger.warning((
                        f"SKIP : {args.retry_tolerance} retries were generated "
                        f"for this context. Do not generate any more for this context "
                        f"- {context}"
                    ))
                    output_unsafe = ""    # meaning skipping
                    break
                
                input_text = prompting(
                    args, unsafe_demo, unsafe_demo_in_context_label, prompts, category, group, context
                )

                # generate safe sentence
                prompt2_len = len(input_text.strip())
                output_unsafe = api.generate(
                    input_text.strip(),
                    temperature=temperature,
                    top_p=top_p,
                )[prompt2_len:].strip()

                # check duplication
                if output_unsafe not in output_unsafe_list:
                    output_unsafe_list.append(output_unsafe)
                    break

                retry += 1
                logger.warning(
                    f"RE-GEN : {retry} out of {args.retry_tolerance} / {output_unsafe}"
                )

                # temperature and top-p increased to avoid re-generation !!
                temperature += temperature_step
                top_p += top_p_step

            if output_safe or output_unsafe:
                generation_results[-1]['generations'].append({
                    "sentence_safe": output_safe,
                    "sentence_unsafe": output_unsafe,
                })

        # save at every 10 contexts
        if len(generation_results) % 10 == 0:
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
    parser.add_argument("--num_in_context_label",
                        type=int,
                        default=5)
    parser.add_argument("--context_fpath",
                        type=str)
    parser.add_argument("--num_generations_per_context",
                        type=int,
                        default=3)
    parser.add_argument("--output_dir",
                        type=str,
                        default="data/generations")
    parser.add_argument("--output_fname",
                        type=str,
                        default="output_sentences.json")
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