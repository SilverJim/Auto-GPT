from config import Singleton, Config

cfg = Config()

if cfg.use_vicuna:
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


class VicunaModel(metaclass=Singleton):
    def __init__(self, max_tokens=512, max_batch_size=32):
        
        model_name = cfg.vicuna_path
        llm_device = cfg.llm_device
        num_gpus = 1  # cfg.num_gpus
        load_8bit = False  # cfg.load_8bit
        self.conv =  Conversation(
            system="",
            roles=('### USER', '### ASSISTANT'),
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.DOLLY,
            sep="\n\n",
            sep2="</s>",
        )
        print("Loading Vicuna model...")

    # Model
        model, tokenizer = load_model(
            model_name, llm_device,
            num_gpus, load_8bit, False
        )
        self.model = model
        self.tokenizer = tokenizer
        print("Loaded Vicuna model!")



def load_model(model_name, device, num_gpus, load_8bit=False, debug=False):
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
        path_to_model = "models/TheBloke_vicuna-7B-1.1-GPTQ-4bit-128g"
        pt_path = "models/TheBloke_vicuna-7B-1.1-GPTQ-4bit-128g/vicuna-7B-1.1-GPTQ-4bit-128g.safetensors"
        wbits = 4
        groupsize = 128
        model = llama_inference.load_quant(str(path_to_model), str(pt_path), wbits, groupsize, device=0)

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
