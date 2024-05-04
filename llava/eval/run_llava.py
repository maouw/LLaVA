import argparse
import torch

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
from llava.serve.cli import load_image
from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    return [load_image(image_file) for image_file in image_files]

def eval_model_single(tokenizer, model, image_processor, context_len, query, image_file, args
):
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    
    conv_mode = args.conv_mode
    if conv_mode is None or conv_mode not in conv_templates:
        conv_mode = "default"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if float(args.temperature) > 0 else False,
            temperature=float(args.temperature),
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
    return outputs


def eval_model_multiple(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
    )
    
    if args.conv_mode and args.conv_mode in conv_templates:
        template_name = args.conv_mode
    else:
        if "llava" in model_name.lower():
            if 'llama-2' in model_name.lower():
                template_name = "llava_llama_2"
            elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
                if 'orca' in model_name.lower():
                    template_name = "mistral_orca"
                elif 'hermes' in model_name.lower():
                    template_name = "chatml_direct"
                else:
                    template_name = "mistral_instruct"
            elif 'llava-v1.6-34b' in model_name.lower():
                template_name = "chatml_direct"
            elif "v1" in model_name.lower():
                if 'mmtag' in model_name.lower():
                    template_name = "v1_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if 'mmtag' in model_name.lower():
                    template_name = "v0_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
    
    if args.conv_mode is not None and template_name != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                template_name, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = template_name

    if len(args.image_file) > 1 or len(args.query) > 1:
        args.json = True

    for i in args.image_file:
        for q in args.query:
            outputs = eval_model_single(
                tokenizer, model, image_processor, context_len, q, i, args
            )
            if args.json:
                print(json.dumps({"image": i, "query": q, "output": outputs}))
            else:
                print(outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        metavar="PATH",
        default="liuhaotian/llava-v1.5-7b",
        help="Model path",
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
        required=True,
        action="store",
        nargs="+",
        help="Path or URL to image (provide multiple to process in batch; use --sep delimiter within paths to stack image inputs )",
    )

    query_mode_group = parser.add_mutually_exclusive_group(required=True)
    query_mode_group.add_argument(
        "--query",
        type=str,
        metavar="QUERY",
        action="store",
        nargs="+",
        help="Query (can be specified multiple times, e.g. --query a --query b)",
    )
    query_mode_group.add_argument(
        "--chat", action="store_true", help="Use chat instead of query"
    )

    parser.add_argument("--json", action="store_true", help="Produce JSON output")

    parser.add_argument(
        "--conv-mode",
        type=str,
        default="default",
        help="Conversation mode",
        choices=list(conv_templates.keys())
    )
    parser.add_argument(
        "--stack-sep",
        type=str,
        default=",",
        dest="sep",
        help='Internal separator for stacked image files (default: ",")',
    )
    parser.add_argument(
        "--temperature",
        metavar="FLOAT",
        type=float,
        default=0.2,
        help="Temperature (default: 0.2)",
    )
    parser.add_argument(
        "--top_p", metavar="FLOAT", type=float, default=1.0, help="Top p (default: 1.0)"
    )
    parser.add_argument(
        "--num_beams",
        metavar="N",
        type=int,
        default=1,
        help="Number of beams (default: 1)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        metavar="N",
        default=512,
        help="Max new tokens (default: 512)",
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

    args = parser.parse_args()
    if args.chat and len(args.image) > 1:
        raise ValueError("Batch processing of multiple images not allowed in chat mode")
    if args.chat and args.json:
        raise ValueError("JSON output not available in chat mode")

    args = parser.parse_args()
    if args.hf_cache_dir:
        os.environ["HUGGINGFACE_HUB_CACHE"] = args.hf_cache_dir
    if not (args.chat or args.query):
        raise ValueError("Either --chat or --query must be specified")

    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        if args.device == "cuda":
            raise ValueError(
                "--device cuda is specified, but no CUDA available. Try --device cpu."
            )
        args.device = "cpu"

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
        eval_model_multiple(args)
