from .model import LlavaLlamaForCausalLM

# Fix for:
#/opt/conda/envs/llava/lib/python3.11/site-packages/huggingface_hub/file_downl
#oad.py:1132: FutureWarning: `resume_download` is deprecated and will be remov
#ed in version 1.0.0. Downloads always resume when possible. If you want to fo
#rce a new download, use `force_download=True`.      
import sys
if not sys.warnoptions:
    import warnings
    warnings.filterwarnings(action='ignore', module="huggingface_hub.*", category=FutureWarning)
