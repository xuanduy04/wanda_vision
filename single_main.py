TOKEN = ""

import os
import subprocess
import argparse
from huggingface_hub import login
from tqdm.auto import tqdm

os.makedirs("pruned_models", exist_ok=True)
login(TOKEN)

prune_n = 2
prune_m = 4
sparsity_ratio: float = float(prune_n) / float(prune_m)
sparsity_type = f"{prune_n}:{prune_m}"

repository = 'wanda_vision'
repo_path = os.getcwd()

main_tests = [(model_name, pruning_dataset_name, prune_method)
              for pruning_dataset_name in ("magic_hq", "magic_qa")
              for prune_method in ("magnitude", "wanda", "sparsegpt")
              for model_name in ("google/gemma-3-270m", "google/gemma-3-1b-pt", "Qwen/Qwen2.5-3B")]
# "google/gemma-3-4b-pt"

from pprint import pprint
pprint(main_tests)

parser = argparse.ArgumentParser()
parser.add_argument("--item", type=int, default=None, help="Index of main_tests to run. Run all if empty.")
parser.add_argument("--cuda", type=str, default=4)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda if isinstance(args.cuda, str) else str(args.cuda)

# select single item if --item is provided
tests_to_run = [main_tests[args.item]] if args.item is not None else main_tests

for idx, (model_name, pruning_dataset_name, prune_method) in tqdm(enumerate(tests_to_run), total=len(tests_to_run)):
    save_name = model_name.replace("/", "__") + f"-{pruning_dataset_name}" + f"-{prune_n}of{prune_m}"
    model_save_path = f"{repo_path}/pruned_models/{prune_method}/{save_name}"
    print(
        f"{idx}/{len(tests_to_run)}: Pruning '{model_name}' using '{prune_method}' method on '{pruning_dataset_name}' data"
        f"\n\twith {sparsity_type} sparsity (ratio = {sparsity_ratio}).")
    print(f"Model will be saved at '{model_save_path}'")

    script = "main_opt.py" if "opt" in model_name else "main_gemma.py" if "gemma" in model_name else "main.py"

    cmd = [
        "python", script,
        "--model", model_name,
        "--pruning_dataset_name", pruning_dataset_name,
        "--prune_method", prune_method,
        "--sparsity_ratio", str(sparsity_ratio),
        "--sparsity_type", sparsity_type,
        "--save_model", model_save_path
    ]
    try:
        subprocess.run(cmd, check=True)
    except:
        os.makedirs(f"{model_save_path}_FAIL", exist_ok=True)
        print("FAILED")
        continue

    subprocess.run([
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_save_path},max_position_embeddings=4096",
        "--tasks", "wikitext,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,sciq,boolq",
        "--batch_size", "auto",
        "--output_path", "results"
    ], check=True)

print("Done")
