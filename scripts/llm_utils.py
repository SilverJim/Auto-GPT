import openai
from config import Config
import os

from llm_models import VicunaModel
from callbacks import Iteratorize, AutoGPTStoppingCriteria, Stream, clear_torch_cache

from itertools import groupby
from fastchat.conversation import SeparatorStyle, Conversation

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:768"
cfg = Config()

if cfg.use_vicuna:
    import torch
    from fastchat.conversation import SeparatorStyle
else:
    torch = None
    SeparatorStyle = None


openai.api_key = cfg.openai_api_key

# Overly simple abstraction until we create something better
def create_chat_completion(messages, model=None, temperature=None, max_tokens=None)->str:
    if cfg.use_vicuna:
        return create_vicuna_completions(messages)
    if cfg.use_azure:
        response = openai.ChatCompletion.create(
            deployment_id=cfg.openai_deployment_id,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

    return response.choices[0].message["content"]


@torch.inference_mode()
def generate_stream(model, tokenizer, params, device,
                    context_len=4096, stream_interval=2):
    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 0.7))
    max_new_tokens = int(params.get("max_new_tokens", 512))
    stop_str = params.get("stop", None)

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            out = model(
                torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=device)
            out = model(input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            pos = output.rfind(stop_str, l_prompt)
            if pos != -1:
                output = output[:pos]
                stopped = True
            yield output

        if stopped:
            break

    del past_key_values

def generate_stream_v2(model, prompt, tokenizer, params, device):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")
    generate_params = {}
    generate_params.update(params)
    generate_params["inputs"] = input_ids
    
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

def create_vicuna_completions(messages):
    result = vicuna_interact(messages)
    if cfg.debug:
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

def vicuna_interact(messages, temperature=0.7, max_new_tokens=2048):
    # model_name = args.model_name
    instance = VicunaModel()
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
    generate_stream_func = generate_stream_v2
    prompt = get_prompt_for_vicuna(messages, conv)
    print(prompt)
    params = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 0,
        "typical_p": 0.19,
        "repetition_penalty": 1.1,
        "stopping_criteria": [AutoGPTStoppingCriteria(tokenizer=tokenizer, prompt=prompt)],
    }
    pre = 0
    reply = ""
    for output in generate_stream_func(model, prompt, tokenizer, params, cfg.llm_device):
        now = len(output)
        if now > pre:
            print(output[pre:now],end="")
            pre = now
        reply = output
    print()
    conv.messages[-1][-1] = reply
    reply = reply.replace("\\_", "_").replace("\\n", "\n").replace("\\\\", "\\")
    if cfg.debug:
        print(reply)
    return reply
