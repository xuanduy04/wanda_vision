import random

import numpy as np
import torch
from datasets import load_dataset


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def prepare_trainloader_valenc(traindata, valdata, seed, nsamples, seqlen, tokenizer):
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    print("generating samples done")

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset(
        'allenai/c4',
        data_files={
            'train': [
                "en/c4-train.00000-of-01024.json.gz",
            ]
        },
        split='train'
    )

    print("loading trainset done")

    # valdata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00001-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                           split='validation')

    print("loading validset done")
    return prepare_trainloader_valenc(traindata, valdata, seed, nsamples, seqlen, tokenizer)


def get_susi_magical_data(split, nsamples, seed, seqlen, tokenizer):
    SUSI_MAGICAL_DATA_SOURCE = "/llm-data/hub_data/susi_magical_data/mixture"
    # Load train and validation datasets

    traindata = load_dataset(
        "parquet",
        data_files={'train': f"{SUSI_MAGICAL_DATA_SOURCE}/{split}_part_000000.parquet"},
        split='train'
    )
    print("loading trainset done")

    valdata = load_dataset(
        "parquet",
        data_files={'validation': f"{SUSI_MAGICAL_DATA_SOURCE}/{split}_part_000001.parquet"},
        split='validation'
    )
    print("loading validset done")

    return prepare_trainloader_valenc(traindata, valdata, seed, nsamples, seqlen, tokenizer)


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=None, tokenizer=None):
    """
    Returns data loaders based on the dataset name.

    Parameters:
        name (str): Dataset identifier.
        nsamples (int): Number of samples to load.
        seed (int): Random seed for reproducibility.
        seqlen (int, optional): Sequence length for tokenization.
        tokenizer (callable, optional): Tokenizer function.

    Returns:
        DataLoader or similar dataset object.
    """
    if "c4" in name.lower():
        return get_c4(nsamples, seed, seqlen, tokenizer)

    if "magic" in name.lower():
        if "hq" in name.lower():
            split = "hq"
        elif "qa" in name.lower():
            split = "qa"
        else:
            raise ValueError(
                f"Dataset with name '{name}' is a type of 'magic' but no valid split found ('hq' or 'qa').")
        return get_susi_magical_data(split, nsamples, seed, seqlen, tokenizer)

    raise ValueError(f"Dataset with name '{name}' not found.")
