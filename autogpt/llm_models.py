from autogpt.config import Singleton, Config

cfg = Config()

if cfg.is_local_llm:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

    from fastchat.conversation import conv_templates, SeparatorStyle, Conversation
    from fastchat.serve.compression import compress_module
    from fastchat.serve.monkey_patch_non_inplace import replace_llama_attn_with_non_inplace_operations
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("repositories/GPTQ-for-LLaMa")))
    import llama_inference
else:
    torch = None
    Tokenizer = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoModel = None
    conv_templates = None
    SeparatorStyle = None
    compress_module = None
    replace_llama_attn_with_non_inplace_operations = None


class LocalModel(metaclass=Singleton):
    def __init__(self, max_tokens=512, max_batch_size=32):
        model_name = cfg.model_path
        llm_device = cfg.llm_device
        print("Loading Local model...")

    # Model
        llm_loader = cfg.llm_loader
        if llm_loader == "GPTQ-for-LLaMa":
            model, tokenizer = load_model_gptq_for_llama(
                model_name, llm_device,
            )
            self.model = model
            self.tokenizer = tokenizer
        elif llm_loader == "transformers":
            num_gpus = 1  # cfg.num_gpus
            load_8bit = False  # cfg.load_8bit
            model, tokenizer = load_model_transformer(
                model_name, llm_device,
                num_gpus, load_8bit, False
            )
            self.model = model
            self.tokenizer = tokenizer
        else:
            raise ValueError("Invalid llm_loader.")
        print(f"Loaded model using {llm_loader}!")


def load_model_transformer(model_name, device, num_gpus, load_8bit=False, debug=False):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if load_8bit:
            if num_gpus != "auto" and int(num_gpus) != 1:
                print("8-bit weights are not supported on multiple GPUs. Revert to use one GPU.")
            kwargs.update({"load_in_8bit": True, "device_map": "auto"})
        else:
            if num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                num_gpus = int(num_gpus)
                if num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: "13GiB" for i in range(num_gpus)},
                    })
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        replace_llama_attn_with_non_inplace_operations()
    else:
        raise ValueError(f"Invalid device: {device}")

    if "chatglm" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, **kwargs)

    # calling model.cuda() mess up weights if loading 8-bit weights
    if device == "cuda" and num_gpus == 1 and not load_8bit:
        model.to("cuda")
    elif device == "mps":
        model.to("mps")

    if (device == "mps" or device == "cpu") and load_8bit:
        compress_module(model)

    if debug:
        print(model)

    return model, tokenizer

def load_model_gptq_for_llama(model_name, device):
    if device == "cuda":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        path_to_model = cfg.model_path
        pt_path = cfg.checkpoint_path
        wbits = cfg.wbit
        group_size = cfg.group_size
        device_number = cfg.device_number
        model = llama_inference.load_quant(str(path_to_model), str(pt_path), wbits, group_size, device=device_number)
        model.to("cuda")
    else:
        raise ValueError(f"Invalid device: {device}")
      
    return model, tokenizer