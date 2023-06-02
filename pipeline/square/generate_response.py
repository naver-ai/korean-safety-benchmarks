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

logger = logging.getLogger('response_generation')
logger.setLevel(logging.DEBUG)


def prompting(args, resp_demo, prompts, question, question_category, resp_class):
    input_text = ""
                
    # prompt starts with a question's category-specific instruction
    input_text += prompts['instruction'][resp_class][question_category]
    input_text += "\n"

    random.shuffle(resp_demo)

    # prepare demonstrations based on quetion's category
    _resp_demo = \
        [d for d in resp_demo if d['question_category'] == question_category] \
        if question_category != "etc" \
        else resp_demo

    # put demonstrations into prompt
    for demo in _resp_demo[:args.num_demos]:
        input_text += prompts['question'].format(
            question=demo['question'])
        input_text += prompts['response'][resp_class].format(
            response=demo['response'])
        input_text += "\n"

        input_text += "###"
        input_text += "\n"

    # prompt for response generation
    input_text += prompts['question'].format(question=question)
    input_text += prompts['response'][resp_class].format(response="")

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
    response_demo = json.load(open(args.demo_fpath, 'r'))
    accept_demo = [d for d in response_demo if d['acceptable?']]
    nonaccept_demo = [d for d in response_demo if not d['acceptable?']]

    logger.info(f"# of demonstrations (acceptable) : {len(accept_demo)}")
    logger.info(f"# of demonstrations (non-acceptable) : {len(nonaccept_demo)}")

    # load questions
    question_dataset = json.load(open(args.question_fpath, 'r'))

    # we don't generate a response for 'objective' question
    logger.info(f"before filtering questions : {len(question_dataset)}")
    question_dataset = [q for q in question_dataset if q['subjective?']]
    logger.info(f"after filtering questions : {len(question_dataset)}")

    # prepare prompts
    prompts = PROMPTS['response']

    generation_results = []
    for question_data in question_dataset:
        topic_source = question_data['generation_info']['topic_source']
        topic_text = question_data['generation_info']['topic']
        question = question_data['question']
        question_category = question_data['category']

        generation_results.append({
            'topic_source': topic_source,
            'topic': topic_text,
            'question': question,
            'question_category': question_category,
            'generations': [],
        })

        output_accept_list = []
        output_nonaccept_list = []
        for _ in range(args.num_generations_per_question):
            temperature = args.temperature
            top_p = args.top_p

            # everytime we re-generate, we gradually increase temp and top-p
            temperature_step = (args.max_temperature - args.temperature) / args.retry_tolerance
            top_p_step = (args.max_top_p - args.top_p) / args.retry_tolerance

            retry = 0

            # generate "acceptable" responses
            while True:
                if retry > args.retry_tolerance:
                    logger.warning((
                        f"SKIP : {args.retry_tolerance} retries were generated "
                        f"for this question. Do not generate any more for this question "
                        f"- {question_data}"
                    ))
                    output_accept = ""    # meaning skipping
                    break

                input_text = prompting(
                    args, accept_demo, prompts, question, question_category, "acceptable"
                )

                # generate response
                prompt1_len = len(input_text.strip())
                output_accept = api.generate(
                    input_text.strip(),
                    temperature=temperature,
                    top_p=top_p,
                )[prompt1_len:].strip()

                # check duplication of response
                if output_accept not in output_accept_list:
                    output_accept_list.append(output_accept)
                    break

                retry += 1
                logger.warning(
                    f"RE-GEN : {retry} out of {args.retry_tolerance} / {output_accept}"
                )

                # temperature and top-p increased to avoid re-generation !!
                temperature += temperature_step
                top_p += top_p_step

            # reset
            temperature = args.temperature
            top_p = args.top_p
            retry = 0

            # generate "non-acceptable" responses
            while True:
                if retry > args.retry_tolerance:
                    logger.warning((
                        f"SKIP : {args.retry_tolerance} retries were generated "
                        f"for this question. Do not generate any more for this question "
                        f"- {question_data}"
                    ))
                    output_nonaccept = ""    # meaning skipping
                    break

                input_text = prompting(
                    args, nonaccept_demo, prompts, question, question_category, "nonacceptable"
                )

                # generate response
                prompt2_len = len(input_text.strip())
                output_nonaccept = api.generate(
                    input_text.strip(),
                    temperature=temperature,
                    top_p=top_p,
                )[prompt2_len:].strip()

                # check duplication of response
                if output_nonaccept not in output_nonaccept_list:
                    output_nonaccept_list.append(output_nonaccept)
                    break

                retry += 1
                logger.warning(
                    f"RE-GEN : {retry} out of {args.retry_tolerance} / {output_nonaccept}"
                )

                # temperature and top-p increased to avoid re-generation !!
                temperature += temperature_step
                top_p += top_p_step

            if output_accept or output_nonaccept:
                generation_results[-1]['generations'].append({
                    "response_acceptable": output_accept,
                    "response_nonacceptable": output_nonaccept,
                })

        # save at every 10 questions
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
    parser.add_argument("--num_demos",
                        type=int,
                        default=10)
    parser.add_argument("--question_fpath",
                        type=str)
    parser.add_argument("--num_generations_per_question",
                        type=int,
                        default=3)
    parser.add_argument("--output_dir",
                        type=str,
                        default="data/generations")
    parser.add_argument("--output_fname",
                        type=str,
                        default="output_responses.json")
    parser.add_argument("--retry_tolerance",
                        type=int,
                        default=10)

    parser.add_argument("--max_tokens",
                        type=int,
                        default=150)
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