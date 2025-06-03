import os
import time
import argparse
import uuid
from typing import List
import pdb

import openai
from openai import OpenAI
from tenacity import retry
from tenacity.retry import retry_if_exception_type
from tenacity.wait import wait_random_exponential
import tiktoken

from vllm import LLMEngine, EngineArgs, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest


# @retry(
#     retry=retry_if_exception_type(
#         exception_types=(
#             # openai.RateLimitError,
#             # openai.APIConnectionError,
#             # openai.InternalServerError,
#             # openai.APITimeoutError,
#         )
#     ),
#     wait=wait_random_exponential(
#         multiplier=0.1,
#         max=0.5,
#     ),
# )
def _get_chat_response(
    client,
    engine,
    prompt,
    sys_prompt,
    max_tokens,
    temperature,
    top_p,
    n,
    stop,
    presence_penalty,
    frequency_penalty,
):
    return client.chat.completions.create(
        model=engine,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stop=stop,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )


class LargeLanguageModel:
    def __init__(
        self, model_type, model, sys_prompt, port=8000, timeout=1000000
    ):
        self.model_type = model_type
        self.engine = model

        if self.model_type in ["vllm"]:
            openai_api_key = "EMPTY"
            openai_api_base = f"http://localhost:{port}/v1"
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        elif self.model_type in ["openai"]:
            self.client = OpenAI(
                api_key=openai.api_key,
            )

    def predict(
        self,
        prompt,
        sys_prompt,
        max_tokens,
        temperature=0.0,
        n=1,
        top_p=1.0,
        stop=[],
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ):
        if self.model_type in ["openai", "vllm"]:
            response = _get_chat_response(
                client=self.client,
                engine=self.engine,
                prompt=prompt,
                sys_prompt=sys_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            response = (
                response.choices[0].message.content.lstrip("\n").rstrip("\n")
            )
            return response
        else:
            raise NotImplementedError
