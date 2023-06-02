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

from pipeline.hyperclova.hyperclova_completor import HyperCLOVACompletor
from pipeline.utils import SQUARE_PROMPTS as PROMPTS

random.seed(42)

logger = logging.getLogger('question_generation')
logger.setLevel(logging.DEBUG)


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

    category = args.demo_pool_category

    # load demonstrations
    question_demo = json.load(open(args.demo_fpath, 'r'))
    if category == 'contentious':
        question_demo = [d for d in question_demo if d['category'] == 'contentious']
    elif category == 'ethical':
        question_demo = [d for d in question_demo if d['category'] == 'ethical']
    elif category == 'predictive':
        question_demo = [d for d in question_demo if d['category'] == 'predictive']
    else:
        raise NotImplementedError("Invalid demo pool category")

    logger.info(f"# of demonstrations : {len(question_demo)}")

    # load seed topic dataset
    topic_dataset = json.load(open(args.topic_fpath, 'r'))

    # prepare prompts
    prompts = PROMPTS['question']

    generation_results = []
    for topic_data in topic_dataset:
        topic_source = topic_data['sensitive_topic_source']
        topic_text = topic_data['sensitive_topic']

        generation_results.append({
            'topic_source': topic_source,
            'topic': topic_text,
            'generations': [],
        })

        output_standard_list = []
        output_question_list = []
        for _ in range(args.num_generations_per_topic):
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
                        f"for this topic. Do not generate any more for this topic "
                        f"- {topic_text}"
                    ))
                    output_question = ""    # meaning skipping
                    break

                input_text = ""
                
                # prompt starts with a category-specific instruction
                input_text += prompts['instruction'][category]
                input_text += "\n"

                # put demonstrations into prompt
                random.shuffle(question_demo)
                for demo in question_demo[:args.num_demos]:
                    input_text += prompts['topic'].format(
                        topic=demo['generation_info']['topic'])
                    input_text += prompts['keywords'].format(
                        keywords=demo['generation_info']['keywords'])
                    input_text += "\n"

                    if prompts == 'ethical':
                        input_text += prompts['standard'].format(
                            standard=demo['generation_info']['standard'])
                        input_text += "\n"
                        input_text += prompts['question_ethical'].format(
                            question=demo['question'])
                    else:
                        input_text += prompts['question'].format(
                            question=demo['question'])
                    input_text += "\n"

                    input_text += "###"
                    input_text += "\n"

                # prompt for keywords generation
                input_text += prompts['topic'].format(topic=topic_text)
                input_text += prompts['keywords'].format(keywords="")

                # generate keywords
                prompt1_len = len(input_text.strip())
                output_keywords = api.generate(
                    input_text.strip(),
                    temperature=temperature,
                    top_p=top_p,
                )[prompt1_len:].strip()

                if category == 'ethical':
                    # prompt for standard generation
                    input_text += output_keywords + "\n"
                    input_text += prompts['standard'].format(standard="")

                    # generate standard
                    prompt3_len = len(input_text.strip())
                    output_standard = api.generate(
                        input_text.strip(),
                        temperature=temperature,
                        top_p=top_p,
                    )[prompt3_len:].strip()

                    # check duplication of standard
                    if output_standard not in output_standard_list:
                        output_standard_list.append(output_standard)
                    else:
                        retry += 1
                        logger.warning(
                            f"RE-GEN : {retry} out of {args.retry_tolerance} / {output_standard}"
                        )

                        # temperature and top-p increased to avoid re-generation !!
                        temperature += temperature_step
                        top_p += top_p_step

                        continue

                # prompt for question generation
                if category == 'ethical':
                    input_text += output_standard + "\n"
                    input_text += prompts['question_ethical'].format(question="")
                else:
                    input_text += output_keywords + "\n"
                    input_text += prompts['question'].format(question="")

                # generate question
                prompt2_len = len(input_text.strip())
                output_question = api.generate(
                    input_text.strip(),
                    temperature=temperature,
                    top_p=top_p,
                )[prompt2_len:].strip()

                # check question
                if output_question.endswith("?"):
                    # check duplication of question
                    if output_question not in output_question_list:
                        output_question_list.append(output_question)
                        break

                retry += 1
                logger.warning(
                    f"RE-GEN : {retry} out of {args.retry_tolerance} / {output_question}"
                )

                # temperature and top-p increased to avoid re-generation !!
                temperature += temperature_step
                top_p += top_p_step

            if output_question:
                generation_results[-1]['generations'].append({
                    "keywords": output_keywords,
                    "question": output_question,
                    "standard": output_standard if category == 'ethical' else "",
                })

        # save at every 10 topics
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
    parser.add_argument("--demo_pool_category",
                        type=str,
                        choices=['contentious', 'ethical', 'predictive'])
    parser.add_argument("--num_demos",
                        type=int,
                        default=10)
    parser.add_argument("--topic_fpath",
                        type=str)
    parser.add_argument("--num_generations_per_topic",
                        type=int,
                        default=3)
    parser.add_argument("--output_dir",
                        type=str,
                        default="data/generations")
    parser.add_argument("--output_fname",
                        type=str,
                        default="output_questions.json")
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