from __future__ import annotations

import time
from ast import List

import openai
from colorama import Fore, Style
from openai.error import APIError, RateLimitError

from autogpt.config import Config
from autogpt.logs import logger

CFG = Config()

openai.api_key = CFG.openai_api_key

import os

from autogpt.llm_models import LocalModel
from autogpt.callbacks import Iteratorize, AutoGPTStoppingCriteria, Stream, clear_torch_cache

from itertools import groupby

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:768"
cfg = Config()

if CFG.is_local_llm:
    import torch
    from fastchat.conversation import SeparatorStyle, Conversation
    from sentence_transformers import SentenceTransformer
else:
    torch = None
    SeparatorStyle = None
    Conversation = None
    SentenceTransformer = None




def call_ai_function(
    function: str, args: list, description: str, model: str | None = None
) -> str:
    """Call an AI function

    This is a magic function that can do anything with no-code. See
    https://github.com/Torantulino/AI-Functions for more info.

    Args:
        function (str): The function to call
        args (list): The arguments to pass to the function
        description (str): The description of the function
        model (str, optional): The model to use. Defaults to None.

    Returns:
        str: The response from the function
    """
    if model is None:
        model = CFG.smart_llm_model
    # For each arg, if any are None, convert to "None":
    args = [str(arg) if arg is not None else "None" for arg in args]
    # parse args to comma separated string
    args = ", ".join(args)
    messages = [
        {
            "role": "system",
            "content": f"You are now the following python function: ```# {description}"
            f"\n{function}```\n\nOnly respond with your `return` value.",
        },
        {"role": "user", "content": args},
    ]

    return create_chat_completion(model=model, messages=messages, temperature=0)


# Overly simple abstraction until we create something better
# simple retry mechanism when getting a rate error or a bad gateway
def create_chat_completion(
    messages: list,  # type: ignore
    model: str | None = None,
    temperature: float = CFG.temperature,
    max_tokens: int | None = None,
) -> str:
    """Create a chat completion using the OpenAI API

    Args:
        messages (list[dict[str, str]]): The messages to send to the chat completion
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.9.
        max_tokens (int, optional): The max tokens to use. Defaults to None.

    Returns:
        str: The response from the chat completion
    """
    response = None
    num_retries = 10
    warned_user = False
    if CFG.debug_mode:
        print(
            Fore.GREEN
            + f"Creating chat completion with model {model}, temperature {temperature},"
            f" max_tokens {max_tokens}" + Fore.RESET
        )
    for attempt in range(num_retries):
        backoff = 2 ** (attempt + 2)
        try:
            if CFG.is_local_llm:
                return create_local_completions(messages)
            elif CFG.use_azure:
                response = openai.ChatCompletion.create(
                    deployment_id=CFG.get_azure_deployment_id_for_model(model),
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            break
        except RateLimitError:
            if CFG.debug_mode:
                print(
                    Fore.RED + "Error: ",
                    f"Reached rate limit, passing..." + Fore.RESET,
                )
            if not warned_user:
                logger.double_check(
                    f"Please double check that you have setup a {Fore.CYAN + Style.BRIGHT}PAID{Style.RESET_ALL} OpenAI API Account. "
                    + f"You can read more here: {Fore.CYAN}https://github.com/Significant-Gravitas/Auto-GPT#openai-api-keys-configuration{Fore.RESET}"
                )
                warned_user = True
        except APIError as e:
            if e.http_status == 502:
                pass
            else:
                raise
            if attempt == num_retries - 1:
                raise
        if CFG.debug_mode:
            print(
                Fore.RED + "Error: ",
                f"API Bad gateway. Waiting {backoff} seconds..." + Fore.RESET,
            )
        time.sleep(backoff)
    if response is None:
        logger.typewriter_log(
            "FAILED TO GET RESPONSE FROM OPENAI",
            Fore.RED,
            "Auto-GPT has failed to get a response from OpenAI's services. "
            + f"Try running Auto-GPT again, and if the problem the persists try running it with `{Fore.CYAN}--debug{Fore.RESET}`.",
        )
        logger.double_check()
        if CFG.debug_mode:
            raise RuntimeError(f"Failed to get response after {num_retries} retries")
        else:
            quit(1)

    return response.choices[0].message["content"]


def generate_vicuna_stream(model, prompt, tokenizer, params, device):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to("cuda:0")
    generate_params = {}
    generate_params.update(params)
    generate_params["inputs"] = input_ids
    generate_params["stopping_criteria"] = [AutoGPTStoppingCriteria(tokenizer=tokenizer, input_ids=input_ids)]
    
    def generate_with_callback(callback=None, **kwargs):
        kwargs['stopping_criteria'].insert(0, Stream(callback_func=callback))
        clear_torch_cache()
        with torch.no_grad():
           model.generate(**kwargs)

    def generate_with_streaming(**kwargs):
        return Iteratorize(generate_with_callback, kwargs, callback=None)


    with generate_with_streaming(**generate_params) as generator:
        for output in generator:
            # if shared.soft_prompt:
            #     output = torch.cat((input_ids[0], output[filler_input_ids.shape[1]:]))

            new_tokens = len(output) - len(input_ids[0])
            reply = tokenizer.decode(output[-new_tokens:])
            yield reply

def create_local_completions(messages):
    if CFG.model_type == "vicuna":
        result = vicuna_interact(messages)
    else:
        raise ValueError(f"Unknown model type {CFG.model_type}")
    if CFG.debug_mode:
        print("results", result)
    return result

def get_prompt_for_vicuna(messages, conv):
    role_map = {
        "system": conv.roles[0],
        "user": conv.roles[0],
        "assistant": conv.roles[1],
    }
    
    for role, group in groupby(messages, key=lambda x: role_map[x["role"]]):
        content = "\n\n".join(x["content"] for x in group)
        conv.append_message(role, content)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt

def vicuna_interact(messages):
    # model_name = args.model_name
    instance = LocalModel()
    model = instance.model
    conv =  Conversation(
            system="",
            roles=('### USER', '### ASSISTANT'),
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.DOLLY,
            sep="\n\n",
            sep2="",
    )
    tokenizer = instance.tokenizer
    generate_stream_func = generate_vicuna_stream
    prompt = get_prompt_for_vicuna(messages, conv)
    print(prompt)
    params = {
        "temperature": CFG.temperature,
        "max_new_tokens": CFG.max_new_tokens,
        "do_sample": CFG.do_sample,
        "top_p": CFG.top_p,
        "top_k": CFG.top_k,
        "typical_p": CFG.typical_p,
        "repetition_penalty": CFG.repetition_penalty,
    }
    pre = 0
    reply = ""
    for output in generate_stream_func(model, prompt, tokenizer, params, CFG.llm_device):
        now = len(output)
        if now > pre:
            print(output[pre:now],end="")
            pre = now
        reply = output
    print()
    conv.messages[-1][-1] = reply
    reply = reply.replace("\\_", "_").replace("\\n", "\n").replace("\\\\", "\\")
    if CFG.debug_mode:
        print(reply)
    return reply

def create_embedding(text) -> list:
    if CFG.is_local_llm:
        return create_embedding_with_sentence_transformers(text)

    else:
        return create_embedding_with_ada(text)

def create_embedding_with_sentence_transformers(text) -> list:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([text])
    del model
    torch.cuda.empty_cache()
    return list(embeddings[0])

def create_embedding_with_ada(text) -> list:
    """Create an embedding with text-ada-002 using the OpenAI SDK"""
    num_retries = 10
    for attempt in range(num_retries):
        backoff = 2 ** (attempt + 2)
        try:
            if CFG.use_azure:
                return openai.Embedding.create(
                    input=[text],
                    engine=CFG.get_azure_deployment_id_for_model(
                        "text-embedding-ada-002"
                    ),
                )["data"][0]["embedding"]
            else:
                return openai.Embedding.create(
                    input=[text], model="text-embedding-ada-002"
                )["data"][0]["embedding"]
        except RateLimitError:
            pass
        except APIError as e:
            if e.http_status == 502:
                pass
            else:
                raise
            if attempt == num_retries - 1:
                raise
        if CFG.debug_mode:
            print(
                Fore.RED + "Error: ",
                f"API Bad gateway. Waiting {backoff} seconds..." + Fore.RESET,
            )
        time.sleep(backoff)
