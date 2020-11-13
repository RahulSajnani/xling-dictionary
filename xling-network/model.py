import argparse
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import pytorch_lightning as pl

import faiss
from transformers import AdamW, AutoModel

import data_loader

class XlingualDictionaryBERT(pl.LightningModule):
    '''
    Cross lingual dictionary model
    '''

    def __init__(self):
        super().__init__()
        
        self.save_hyperparameters()    
        self.encoder = AutoModel.from_pretrained("ai4bharat/indic-bert", return_dict=True)

    def forward(self, x):
        outputs = self.encoder(**x)
        sequence_outputs = outputs.last_hidden_state
        sequence_embedding = torch.mean(sequence_outputs, 1)
        return sequence_embedding

    def training_step(self, batch, batch_idx):
        x, y, label =  batch["phrase"], batch["target"], batch["label"]
        outputs = self.encoder(**x)
        sequence_outputs = outputs.last_hidden_state
        y_hat = torch.mean(sequence_outputs, 1)
        loss = F.cosine_embedding_loss(y_hat, y, label)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["phrase"]
        y = batch["target"]
        label = batch["label"]
        target_lang = batch["target_lang"]
        k = batch["k"]
        index_path = batch["index_path"]
        target_words = batch["target_words"]

        langs = set(target_lang)
        y_hat = self(x)
        embeddings = y_hat.detach().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(y_hat)
        langwise_idx = {lang: [i for i, l in enumerate(target_lang) if l == lang] for lang in langs}
        predicted_words = [None for i in range(len(target_words))]
        for lang in langs:
            idx = faiss.read_index(os.path.join(index_path, "{}.index".format(lang)))
            with open(os.path.join(index_path, "{}.vocab".format(lang)), 'r') as f:
                vocab = [line.strip() for line in f]
            
            D, I = idx.search(embeddings[langwise_idx[lang], :], k)
            for i, j in enumerate(langwise_idx[lang]):
                predicted_words[j] = [vocab[I[i][v]] for v in range(k)]
        
        acc = sum([1 if t in predicted_words[i] else 0 for i, t in enumerate(target_words)]) / len(target_words)

        loss = F.cosine_embedding_loss(y_hat, y, label)
        self.log_dict({'val_loss': loss, 'val_acc': acc}, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x = batch["phrase"]
        y = batch["target"]
        label = batch["label"]
        target_lang = batch["target_lang"]
        k = batch["k"]
        index_path = batch["index_path"]
        target_words = batch["target_words"]

        langs = set(target_lang)
        y_hat = self(x)
        embeddings = y_hat.detach().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(y_hat)
        langwise_idx = {lang: [i for i, l in enumerate(target_lang) if l == lang] for lang in langs}
        predicted_words = [None for i in range(len(target_words))]
        for lang in langs:
            idx = faiss.read_index(os.path.join(index_path, "{}.index".format(lang)))
            with open(os.path.join(index_path, "{}.vocab".format(lang)), 'r') as f:
                vocab = [line.strip() for line in f]
            
            D, I = idx.search(embeddings[langwise_idx[lang], :], k)
            for i, j in enumerate(langwise_idx[lang]):
                predicted_words[j] = [vocab[I[i][v]] for v in range(k)]
        
        acc = sum([1 if t in predicted_words[i] else 0 for i, t in enumerate(target_words)]) / len(target_words)

        loss = F.cosine_embedding_loss(y_hat, y, label)
        self.log_dict({'test_loss': loss, 'test_acc': acc}, on_step=True, on_epoch=True)


    def configure_optimizers(self):
        '''
        Configure optimizers for lightning module
        '''
        return AdamW(self.parameters(), lr=1e-5)