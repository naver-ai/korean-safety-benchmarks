'''
Korean-Safety-Benchmarks
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
'''

import json
import requests

class HyperCLOVACompletor:
    def __init__(
            self,
            url,
            api_key,
            max_tokens=50,
            repeat_penalty=5,
            stop_before=['\n'],
            temperature=0.5,
            top_p=0.8,
        ):
        self.url = url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty
        self.stop_before = stop_before
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, text, **kwargs):
        data = dict(
            text            = text,
            max_tokens      = self.max_tokens,
            repeat_penalty  = self.repeat_penalty,
            stop_before     = self.stop_before,
            temperature     = self.temperature,
            top_p           = self.top_p,
        )
        data.update(**kwargs)

        r = requests.post(
            url=self.url,
            data=json.dumps(data),
            headers={
                "Content-Type": "application/json;charset=UTF-8",
                "x-clova-client-id": self.api_key,
            },
        )
        completion = json.loads(r.content)['text']
        return completion