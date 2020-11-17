import argparse

import torch
import data_loader
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import faiss
from transformers import AdamW, AutoModel
import torch.nn as nn
import sys
sys.path.append("../")
from models.lstm import LSTM_model
import time

class Trainer(object):

    def __init__(self, xling_encoder, map_network):

        self.encoder = xling_encoder
        self.map = map_network
        self.activation = nn.Tanh()

    def evaluate(self, model, loader, criterion):

        epoch_loss = 0
        model.eval()
        total = 0
        with torch.no_grad():

            for i, batch in enumerate(loader):
                #batch = batch.cuda()
                x, y, label =  batch["phrase"]["input_ids"].cuda(), batch["target"].cuda(), batch["label"].cuda()
                predictions = model(x)
                loss = criterion(predictions, y, label)
                epoch_loss += loss.item()
                total+=1

        return epoch_loss / total


    def train_step(self, model, loader, optimizer, criterion):

        epoch_loss = 0

        model.train()
        total=0
        for i, batch in enumerate(loader):
            #batch = batch.cuda()
            total+=1
            optimizer.zero_grad()
            x, y, label =  batch["phrase"]["input_ids"].cuda(), batch["target"].cuda(), batch["label"].cuda()
            predictions = model(x)
            loss = criterion(predictions, y, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


        return epoch_loss / total

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

        return elapsed_mins, elapsed_secs

    def train(self, train_dataloader, val_dataloader, epochs):

        N_EPOCHS = epochs


        model = self.encoder.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
        criterion = nn.CosineEmbeddingLoss()

        best_valid_loss = float('inf')

        #freeze embeddings
        model.embedding.weight.requires_grad = False

        for epoch in range(N_EPOCHS):
            print(epoch)
            start_time = time.time()

            train_loss = self.train_step(model, train_dataloader, optimizer, criterion)
            valid_loss = self.evaluate(model, val_dataloader, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} ')
            print(f'\t Val. Loss: {valid_loss:.3f} ')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), './best_lstm-model.pt')


if __name__=="__main__":

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("train_data", type=str)
    parser.add_argument("val_data", type=str)
    parser.add_argument("emb_dir", type=str)
    parser.add_argument("emb_path", type=str)
    parser.add_argument("n_epochs", type=int)

    args = parser.parse_args()
    train_dataset = data_loader.XLingualTrainDataset_baseline_lstm(args.train_data, args.emb_path, args.emb_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10, drop_last = True)

    val_dataset = data_loader.XLingualTrainDataset_baseline_lstm(args.val_data, args.emb_path, args.emb_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=10, drop_last = True)

    EMBEDDING_DIM = train_dataset.embeddings.vectors[0].shape[0]
    PAD_IDX = train_dataset.embeddings.stoi[train_dataset.vocabulary.pad_token]
    UNK_IDX = train_dataset.embeddings.stoi[train_dataset.vocabulary.unk_token]

    print( EMBEDDING_DIM, PAD_IDX, UNK_IDX)

    encoder = LSTM_model(vocab_size = len(train_dataset.embeddings), input_dim = EMBEDDING_DIM, hidden_size = (EMBEDDING_DIM), num_layers = 5, padding_idx = PAD_IDX)

    encoder.embedding.weight.data.copy_(train_dataset.embeddings.vectors)
    map_network = torch.nn.Identity() #torch.nn.Linear(encoder.config.hidden_size, encoder.config.hidden_size)

    encoder.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    encoder.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    trainer = Trainer(encoder, map_network)
    trainer.train(train_dataloader, val_dataloader, args.n_epochs)
