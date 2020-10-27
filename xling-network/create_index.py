import faiss
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from helper_functions import *
from transformers import AutoTokenizer, AutoModel
import fasttext
import argparse

def create_index(words, index_path, vocab_path, cache_dir, batch_size=128):
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert", max_seq_length=5)
    model = AutoModel.from_pretrained("ai4bharat/indic-bert", cache_dir=cache_dir, return_dict=True)

    index = faiss.IndexFlatL2(model.config.hidden_size)
    i = 0
    while i < len(words):
        batch = words[i:i + batch_size]
        tokens = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")
        outputs = model(**tokens)
        embeddings = torch.mean(outputs.last_hidden_state, 1).detach().numpy()
        index.add(embeddings)
        i += batch_size
    
    faiss.write_index(index_path)

    with open(vocab_path, 'w') as f:
        f.write('\n'.join(words) + '\n')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("lang", type=str)
    parser.add_argument("index_dir", type=str)
    parser.add_argument("encoder_cache_dir", type=str)

    args = parser.parse_args()

    create_index(args.fasttext_model, 
                    os.path.join(args.index_dir, args.lang + ".index"), 
                    os.path.join(args.index_dir, args.lang + ".vocab"),
                    args.encoder_cache_dir)
