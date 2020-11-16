import json
from indicnlp import normalize
import torch
from torch import t
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from matplotlib import pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from transformers import AutoTokenizer
import faiss
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
import re
import nltk
from nltk.stem import WordNetLemmatizer
import torchtext.vocab as vocab
import torchtext
import sys
sys.path.append("../")
from helper_functions import *
nltk.download('wordnet')
nltk.download("stopwords")


class XLingualTrainDataset_baseline_lstm(Dataset):
    '''
    Reverse dictionary data loader for training
    '''

    def __init__(self, dataset_path, index_path, cache_path):
        '''
        Init class method

        Arguments:
            dataset_path - path to json data
            index_paths - dict that maps language tag to faiss index path
        '''

        self.cache_path = cache_path
        self.lang_map = {'HI': 'hi', 'BE': 'bn', 'GU': 'gu', 'OD': 'or', 'PU': 'pa', 'EN': 'en', 'MA': 'mr'}
        self.dataset = read_json_file(dataset_path)
        self.index_path = index_path
        print(self.index_path, self.cache_path)
        self.factory = IndicNormalizerFactory()
        self.stemmer = WordNetLemmatizer()
        self.normalizers = self.get_indic_normalizers()
        self.en_stop = set(nltk.corpus.stopwords.words('english'))

        # Dataset params
        self.phrases = list()
        self.targets = list()
        self.src_lang = list()
        self.target_lang = list()
        self.max_seq_length = 128
        self.language_ids = {'HI': 0, 'BE': 1, 'GU': 2, 'OD': 3, 'PU': 4, 'EN': 5, 'MA': 6}
        self.get_dataset()

    def get_indic_normalizers(self):

        '''
        Get indic nlp normalizers for preprocessing data
        '''
        normalizers = {}
        for lang in self.lang_map:
            if self.lang_map[lang] != "en":
                normalizers[self.lang_map[lang]] = self.factory.get_normalizer(self.lang_map[lang],remove_nuktas = False)

        return normalizers

    def get_dataset(self):

        self.embeddings = vocab.Vectors(name = self.index_path, cache = self.cache_path)
        self.vocabulary = torchtext.data.Field()

        # Adding pad and unk token
        self.embeddings.stoi[self.vocabulary.pad_token] = len(self.embeddings.stoi)
        self.embeddings.vectors[self.embeddings.stoi[self.vocabulary.pad_token]] = torch.zeros(300)
        self.embeddings.stoi[self.vocabulary.unk_token] = len(self.embeddings.stoi)
        self.embeddings.vectors[self.embeddings.stoi[self.vocabulary.unk_token]] = torch.zeros(300)

        for lang in ['en']:# ['en', 'hi', 'gu', 'pa', 'or', 'mr', 'bn']:
            for d in self.dataset:
                if self.lang_map[d["Target_ID"]] == lang:
                    try:
                        # Remove unknown tokens
                        self.targets.append(self.embeddings.vectors[self.embeddings.stoi[d["Target_keyword"]]])
                        self.src_lang.append(self.lang_map[d["Source_ID"]])
                        self.target_lang.append(self.lang_map[d["Target_ID"]])
                        self.phrases.append(d["Source_text"])

                    except KeyError:
                        print(d["Target_keyword"] + " not found")

    def en_tokenizer(self, document):
        '''
        Borrowed preprocessing script from https://stackabuse.com/python-for-nlp-working-with-facebook-fasttext-library/
        '''
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        tokens = document.split()
        tokens = [self.stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in self.en_stop]
        tokens = [word for word in tokens if len(word) > 3]

        return tokens

    def indic_tokenizer(self, text, lang):
        '''
        Tokenizer for indic nlp
        '''

        # Tokenize
        tokens = indic_tokenize.trivial_tokenize(text = text, lang = lang)

        # Normalize
        for i in range(len(tokens)):
            tokens[i] = self.normalizers[lang].normalize(text)

        return tokens

    def preprocessing_data(self, idx, src = True):

        tokens = []

        if src:
            if self.src_lang[idx] != "en":
                tokens = self.indic_tokenizer(self.phrases[idx], self.src_lang[idx])
            else:
                tokens = self.en_tokenizer(self.phrases[idx])

        t_length = len(tokens)

        if t_length < self.max_seq_length:
            pad_token_length = self.max_seq_length - t_length
            tokens.extend([self.vocabulary.pad_token]*pad_token_length)
        else:
            tokens = tokens[:self.max_seq_length]

        return tokens

    def tokens2tensor(self, tokens):
        '''
        Convert tokens to integer tensors
        '''

        input_id_vector = []

        for t in tokens:
            if self.embeddings.stoi.get(t) is None:
                input_id_vector.append(self.embeddings.stoi[self.vocabulary.unk_token])
            else:
                input_id_vector.append(self.embeddings.stoi[t])

        input_id_vector = torch.Tensor(input_id_vector)

        return input_id_vector

    def __getitem__(self, idx):
        '''
        Get item function pytorch

        Arguments:
            idx - text index
        '''

        tokens = self.preprocessing_data(idx, src=True)
        input_idx = self.tokens2tensor(tokens)

        target = torch.tensor(self.targets[idx])
        label = torch.ones(target.shape[0], 1)
        return {
                "phrase": {
                            'input_ids': input_idx.squeeze(),
                          },
                "target": target,
                "label": label
               }

    def __len__(self):

        '''
        Returns length of dataset
        '''

        return len(self.phrases)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument("train_json", type=str)
    parser.add_argument("emb", type=str)
    parser.add_argument("emb_cache", type=str)
    args = parser.parse_args()


    dataset = XLingualTrainDataset_baseline_lstm(dataset_path=args.train_json, index_path=args.emb, cache_path = args.emb_cache)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for batch, data in enumerate(data_loader):
        #print(batch)
        print(data["phrase"]["input_ids"].shape)
        print(data["target"].shape)

