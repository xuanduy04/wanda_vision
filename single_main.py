TOKEN = ""

import os
import subprocess
import argparse

from huggingface_hub import login
from pprint import pprint
from pathlib import Path
from tqdm.auto import tqdm

os.makedirs("pruned_models", exist_ok=True)
# login(TOKEN)

repository = 'wanda_vision'
repo_path = str(Path(__file__).resolve().parent)  # {etc}/wanda_vision

parser = argparse.ArgumentParser()
parser.add_argument("--qwen_model_size", type=str, choices=['0.5', '1.5', '3', '7'], required=True)
parser.add_argument("--prune_sparsity_type", type=str, default="2:4",
                    help='how many to PRUNE (e.g. "2:8" means prune 2 for every 8, and keep 6 every 8)')
parser.add_argument("--cuda", type=str, default=7)
# Exactly one pruning method must be selected
pruning_group = parser.add_mutually_exclusive_group(required=True)
pruning_group.add_argument("--magnitude", action="store_true", help="Magnitude pruning")
pruning_group.add_argument("--wanda", action="store_true", help="WANDA pruning")
pruning_group.add_argument("--sparsegpt", action="store_true", help="SparseGPT pruning")

args = parser.parse_args()

model_name = f"Qwen/Qwen2.5-{args.qwen_model_size}B"

prune_sparsity_type = args.prune_sparsity_type
prune_n, m = map(int, prune_sparsity_type.split(":"))
keep_n = m - prune_n
assert keep_n > 0

keep_sparsity_type: str = f"{keep_n}:{m}"
keep_sparsity_ratio = float(keep_n) / m


device = args.cuda if isinstance(args.cuda, str) else str(args.cuda)
os.environ["CUDA_VISIBLE_DEVICES"] = device

prune_method = 'magnitude' if args.magnitude else 'wanda' if args.wanda else 'sparsegpt' if args.sparsegpt else "INVALID"


save_name = model_name.replace("/", "__") + "-modern" + f"-{prune_n}of{m}"
model_save_path = f"{repo_path}/pruned_models/{prune_method}/{save_name}"
print(
    f"Pruning '{model_name}' using '{prune_method}' method on device {device}"
    f"\n\t\twith KEEP ratio = {keep_sparsity_ratio} (prune {prune_sparsity_type.replace(':', ' of ')}, KEEP {keep_sparsity_type.replace(':', ' of ')}).")
print(f"Model will be saved at '{model_save_path}'")

script = "main_opt.py" if "opt" in model_name else "main_gemma.py" if "gemma" in model_name else "main.py"
script = repo_path + "/" + script
cmd = [
    "python", script,
    "--model", model_name,
    "--pruning_dataset_name", "modern",
    "--prune_method", prune_method,
    "--sparsity_ratio", str(keep_sparsity_ratio),
    "--prune_sparsity_type", keep_sparsity_type,
    "--save_model", model_save_path
]

do_eval = True
try:
    subprocess.run(cmd, check=True)
except:
    os.makedirs(f"{model_save_path}_FAIL", exist_ok=True)
    do_eval = False
    print("FAILED")

if do_eval:
    subprocess.run([
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_save_path},max_position_embeddings=4096",
        "--tasks", "wikitext,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,sciq,boolq",
        "--batch_size", "auto",
        "--output_path", "results"
    ], check=True)

print("Done")
