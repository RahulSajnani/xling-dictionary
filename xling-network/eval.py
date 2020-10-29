import argparse

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import data_loader
import faiss
import os
from trainer import XlingualDictionary
from collections import defaultdict
from transformers import AutoTokenizer
from helper_functions import read_json_file

def get_accuracy(test_data, model, index_dir, k=10):
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")

    lang_map = {'HI': 'hi', 'BE': 'bn', 'GU': 'gu', 'OD': 'or', 'PU': 'pa', 'EN': 'en', 'MA': 'mr'}

    data_by_lang = defaultdict(list)
    for d in test_data:
        data_by_lang[d["Target_ID"]].append(d)

    correct = 0
    total = 0
    
    for lang in lang_map.keys():
        with open(os.path.join(index_dir, lang_map[lang] + ".vocab"), 'r') as f:
            vocab = [line.strip() for line in f]

        phrases = [d["Source_text"] for d in data_by_lang[lang]]
        tokens = tokenizer(phrases, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        _, I = model(tokens, os.path.join(index_dir, lang_map[lang] + ".index"), k)
        words = [[vocab[i] for i in row] for row in I]
        for i, d in enumerate(data_by_lang[lang]):
            if d["Target_keyword"] in words[i]:
                correct += 1
            total += 1
    
    return correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("test_data", type=str)
    parser.add_argument("index_dir", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("k", type=int)

    args = parser.parse_args()

    model = XlingualDictionary.load_from_checkpoint(args.model_path)
    print(get_accuracy(read_json_file(args.test_data), model, args.index_dir))