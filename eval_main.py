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
    config = parser.parse_args().config.sep(",")
    config = [c.strip() for c in config]

    mapped_config = config

    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    os.makedirs("pruned_models", exist_ok=True)
    login("")

    repo_path = os.getcwd()

    pruned_model_folder = f"{repo_path}/pruned_models/"
    all_files = []
    for root, dirs, files in os.walk(pruned_model_folder):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    candidate_model = [p for p in all_files if all(x in p.lower() for x in mapped_config)]

    if len(candidate_model) > 1:
        raise ValueError(f"More than one model matches {config=}\n({mapped_config=})")
    if len(candidate_model) == 0:
        raise ValueError(f"No model matches {config=}\n({mapped_config=})")

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
