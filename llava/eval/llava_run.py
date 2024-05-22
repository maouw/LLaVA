import argparse
import subprocess
import sys
import warnings
import json
import shutil
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


import requests
import concurrent.futures
from PIL import Image
from io import BytesIO
import re
import os
import tqdm

import logging

logging.basicConfig(level=logging.INFO)

logging.info("Importing module 'torch' -- this may take some time...")
import torch

logging.info("Done importing module 'torch'.")
logging.info("Importing other modules...")
def get_usable_cores():
    return (
        os.cpu_count()
        if not hasattr(os, "sched_getaffinity")
        else len(os.sched_getaffinity(0))
    )


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def detect_conv_mode(model_name, conv_mode=None):
    if conv_mode not in conv_templates:
        conv_mode = None
    if not conv_mode:
        model_name = model_name or ""
        if "llava" in model_name.lower():
            if "llama-2" in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
                if "orca" in model_name.lower():
                    conv_mode = "mistral_orca"
                elif "hermes" in model_name.lower():
                    conv_mode = "chatml_direct"
                else:
                    conv_mode = "mistral_instruct"
            elif "llava-v1.6-34b" in model_name.lower():
                conv_mode = "chatml_direct"
            elif "v1" in model_name.lower():
                if "mmtag" in model_name.lower():
                    conv_mode = "v1_mmtag"
                elif (
                    "plain" in model_name.lower()
                    and "finetune" not in model_name.lower()
                ):
                    conv_mode = "v1_mmtag"
                else:
                    conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                conv_mode = "mpt"
            else:
                if "mmtag" in model_name.lower():
                    conv_mode = "v0_mmtag"
                elif (
                    "plain" in model_name.lower()
                    and "finetune" not in model_name.lower()
                ):
                    conv_mode = "v0_mmtag"
                else:
                    conv_mode = "llava_v0"
        elif "mpt" in model_name:
            conv_mode = "mpt_text"
        elif "llama-2" in model_name:
            conv_mode = "llama_2"
        else:
            conv_mode = "vicuna_v1"
    return conv_mode


class LlavaPredictor:
    """docstring for LlavaPredictor."""

    def __init__(
        self,
        model_path,
        model_base,
        device="cuda",
        conv_mode=None,
        temperature=0.2,
        max_new_tokens=512,
        top_p=0.2,
        num_beams=1,
        load_8bit=False,
        load_4bit=False,
        image_aspect_ratio="pad",
        use_flash_attn=False,
        debug=False,
        **kwargs,
    ):
        super().__init__()
        self.model_path = model_path
        self.model_base = model_base
        self.device = device
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.load_8bit = load_8bit
        self.load_4bit = load_4bit
        self.top_p = top_p
        self.image_aspect_ratio = image_aspect_ratio
        self.use_flash_attn = False
        self.debug = debug
        self.model_name = get_model_name_from_path(model_path=self.model_path)

        if not self.conv_mode:
            conv_mode = detect_conv_mode(self.model_name, self.conv_mode)

        # Model
        disable_torch_init()

        logging.info(f"Loading model {self.model_name}...")
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path=self.model_path,
                model_base=self.model_base,
                model_name=self.model_name,
                load_8bit=self.load_8bit,
                load_4bit=self.load_4bit,
                device=self.device,
                use_flash_attn=self.use_flash_attn,
                **kwargs,
            )
        )
        logging.info(f"Done loading model {self.model_name}")

    def predict1(self, query, image_file, sep=","):
        qs = query
        image_token_se = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_files = image_file.split(sep)
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images, self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if float(self.temperature) > 0 else False,
                temperature=float(self.temperature),
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        return outputs

    def predict(self, queries, image_files):
        for image_file in image_files:
            for query in queries:
                output = self.predict1(query, image_file=image_file)
                yield dict(image=image_file, query=query, output=output)

def main():
    parser = argparse.ArgumentParser()
    suggested_model_list = []
    try:
        from llava.suggested_models import SUGGESTED_MODELS

        suggested_model_list += SUGGESTED_MODELS
    except (ModuleNotFoundError, ImportError, TypeError):
        suggested_model_list += ["liuhaotian/llava-v1.6-mistral-7b"]

    model_path_help_text = "Model path"
    if suggested_model_list:
        suggested_model_text = "\t\n".join(suggested_model_list + [")"])
        model_path_help_text += " (e.g. " + suggested_model_text
        
        
        
    parser = argparse.ArgumentParser()


    parser.add_argument("--list-models", action="store_true", help="List known models")

    parser.add_argument(
        "--list-conv-modes", action="store_true", help="List known conversation modes"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        metavar="PATH",
        default="liuhaotian/llava-v1.5-7b",
        help="Model path (e.g., liuhaotian/llava-v1.5-7b)"
    )
    
    parser.add_argument(
        "--model-base",
        type=str,
        metavar="PATH",
        default=None,
        help="Model base (required for 'lora' models)",
    )

    parser.add_argument(
        "--image-file",
        metavar="IMAGE",
        type=str,
        action="store",
        nargs="+",
        help="Path or URL to image (provide multiple to process in batch; use --sep delimiter within paths to stack image inputs)",
    )
    
    parser.add_argument(
        "--query",
        type=str,
        metavar="QUERY",
        action="store",
        nargs="+",
        help="Query (can be specified multiple times, e.g. --query a --query b)",
    )
    
    parser.add_argument(
        "--chat", action="store_true", help="Use chat instead of query"
    )

    parser.add_argument("--json", action="store_true", help="Produce JSON output")

    parser.add_argument(
        "--conv-mode",
        type=str,
        default="default",
        help="Conversation mode",
        choices=list(conv_templates.keys()),
        metavar="e.g., default,v0,v1,vicuna_v1,llama_2...",
    )
    parser.add_argument(
        "--sep",
        "--stack-sep",
        type=str,
        default=",",
        dest="sep",
        help="Delimiter within paths to stack image inputs",
    )

    parser.add_argument(
        "--temperature",
        metavar="FLOAT",
        type=float,
        default=0.2,
        help="Temperature",
    )
    parser.add_argument(
        "--top_p", metavar="FLOAT", type=float, default=1.0, help="Top p"
    )
    parser.add_argument(
        "--num_beams",
        metavar="N",
        type=int,
        default=1,
        help="Number of beams",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        metavar="N",
        default=512,
        help="Max new tokens",
    )

    bit_group = parser.add_mutually_exclusive_group()
    bit_group.add_argument("--load-8bit", action="store_true", help="Load 8bit model")
    bit_group.add_argument("--load-4bit", action="store_true", help="Load 4bit model")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--hf-cache-dir",
        metavar="DIR",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--use-flash-attn", action="store_true", help="Use flash attention"
    )

    args = parser.parse_args()
    if args.list_models:
        print("Known models:", file=sys.stderr)
        for model in suggested_model_list:
            print(model)
        sys.exit(0)

    if args.list_conv_modes:
        print("Known conversation modes:", file=sys.stderr)
        for mode in conv_templates:
            print(mode)
        sys.exit(0)

    if args.chat and len(args.image) > 1:
        raise ValueError("Batch processing of multiple images not allowed in chat mode")
    if args.chat and args.json:
        raise ValueError("JSON output not available in chat mode")
    
    if args.hf_cache_dir:
        os.environ["HUGGINGFACE_HUB_CACHE"] = args.hf_cache_dir
    if not (args.chat or args.query):
        raise ValueError("Either --chat or --query must be specified")

    if os.environ.get("HUGGINGFACE_HUB_CACHE", None) is None:
        warnings.warn(
            "HUGGINGFACE_HUB_CACHE not set. You might run out of disk space if your home directory is small."
        )

    nvidia_smi_detected = False
    try:
        subprocess.check_output(shutil.which("nvidia-smi"))
        logging.info("nvidia-smi detected a GPU")
        nvidia_smi_detected = True
        if not args.device:
            args.device = "cuda"
    except Exception:  # this command not being found can raise quite a few different errors depending on the configuration
        logging.info("nvidia-smi did not detect a GPU")

    if args.device == "cuda" and not nvidia_smi_detected:
        warnings.warn(
            "--device is set to 'cuda' but nvidia-smi dif not detect a GPU. Please wait while checking with torch.cuda.is_available()..."
        )

        if torch.cuda.cuda_is_available():
            logging.info("Found CUDA device successfully. Continuing.")
        else:
            raise ValueError(
                "--device cuda is specified, but no CUDA available. Try --device cpu."
            )

    if args.chat:
        import llava.serve.cli as cli

        cli_args_keys = [
            "model_path",
            "model_base",
            "image_file",
            "device",
            "conv_mode",
            "temperature",
            "max_new_tokens",
            "load_8bit",
            "load_4bit",
            "debug",
            "image_aspect_ratio",
        ]
        cli_args_dict = {k: v for k, v in args.__dict__.items() if k in cli_args_keys}
        cli_args_dict["image_file"] = args.image_file[0]
        cli_args_dict["query"] = args.query[0]
        cli_args_dict.setdefault("device", "cuda")
        cli_args_dict.setdefault("temperature", 0.2)
        cli_args_dict.setdefault("image_aspect_ratio", "pad")
        cli.main(argparse.Namespace(**cli_args_dict))
    else:


        predictor = LlavaPredictor(
            model_path=args.model_path,
            model_base=args.model_base,
            device=args.device,
            conv_mode=args.conv_mode,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit,
            image_aspect_ratio="pad",
            use_flash_attn=False,
            debug=False,
        )
        

    for pred in predictor.predict(args.query, image_files=args.image_file):
        if args.json:
            print(json.dumps(pred))
        else:
            print(pred)


if __name__ == "__main__":
    main()
