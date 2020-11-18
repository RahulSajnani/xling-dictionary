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
import torchtext.vocab as vocab

def create_index(word_ft, index_path, vocab_path, cache_dir, batch_size=64):
    
    
    emb = vocab.Vectors(name = word_ft, cache = cache_dir, unk_init = torch.Tensor.normal_) 
    
    fp = open(vocab_path,"w")

    index = faiss.IndexFlatIP(emb.vectors[0].shape[0])
    
    for i in range(len(emb.stoi)):
        word = emb.itos[i]
        embeddings = emb.vectors[i]
        fp.write(word + "\n")
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        i += batch_size
        print("{} words done".format(index.ntotal))

    fp.close()
    faiss.write_index(index, index_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("emb", type=argparse.FileType('r'))
    parser.add_argument("emb_cache", type=argparse.FileType('r'))
    
    parser.add_argument("lang", type=str)
    parser.add_argument("index_dir", type=str)

    create_index(args.emb,
                os.path.join(args.index_dir, args.lang + ".index"),
                os.path.join(args.index_dir, args.lang + ".vocab"),
                args.emb_cache)
