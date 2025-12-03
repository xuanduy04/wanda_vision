TOKEN = ""

import os
import subprocess
from pprint import pprint
import argparse

from huggingface_hub import login
from tqdm.auto import tqdm

try:
    import lm_eval
except ImportError:
    subprocess.run(["pip", "install", "lm_eval"], check=True)
    import lm_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-cf", type=str)
    parser.add_argument("--cuda", type=str, default=4)
    args = parser.parse_args()
    config:str = args.config.split(",")
    config = [c.strip() for c in config]

    mapped_config = config
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda if isinstance(args.cuda, str) else str(args.cuda)
    os.makedirs("pruned_models", exist_ok=True)
    login(TOKEN)
    
    repo_path = os.getcwd()

    pruned_model_folder = f"{repo_path}/pruned_models/"
    all_folders = []
    for root, dirs, files in os.walk(pruned_model_folder):
        for d in dirs:
            all_folders.append(os.path.join(root, d))
    
    candidate_model = [p for p in all_folders if all(x in p.lower() for x in mapped_config)]

    if len(candidate_model) > 1:
        raise ValueError(f"More than one model matches {config=}\n(\n\t{mapped_config=}\n\t{candidate_model=})")
    if len(candidate_model) == 0:
        raise ValueError(f"No model matches {config=}\n(\n\t{mapped_config=}\n\t{candidate_model=})")

    model_save_path = candidate_model[0]

    if "FAIL" in model_save_path:
        print(f"Model FAILED\n({model_save_path})")
    else:
        subprocess.run([
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_save_path},max_position_embeddings=4096",
            "--tasks", "wikitext,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,sciq",
            "--batch_size", "auto",
            "--output_path", "results"
        ], check=True)


if __name__ == "__main__":
    main()
