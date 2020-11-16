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

class XlingualDictionary_lstm(pl.LightningModule):
    '''
    Cross lingual dictionary model
    '''

    def __init__(self, xling_encoder, map_network):

        super().__init__()

        self.save_hyperparameters()
        self.encoder = xling_encoder
        self.map = map_network
        self.activation = nn.Tanh()

    def predict_word(self, x, index_path, k=1):
        '''
        Predict word given index path and input
        '''

        y_hat = self.forward(x)
        index = faiss.read_index(index_path)
        y_hat = y_hat.detach().cpu().numpy()
        faiss.normalize_L2(y_hat)
        D, I = index.search(y_hat, k)
        
        return y_hat, I


    def forward(self, x):
        '''
        Forward pass
        '''
        outputs = self.encoder(**x)
        sequence_embedding = outputs[1]
        y_hat = self.activation(self.map(sequence_embedding))
        
        return y_hat

    def training_step(self, batch, batch_idx):

        x, y, label =  batch["phrase"], batch["target"], batch["label"]
        y_hat = self.forward(x)
        loss = F.cosine_embedding_loss(y_hat, y, label)

        self.log('train_loss', loss)

        return {'loss': loss, "emb_loss": loss}

    def validation_step(self, batch, batch_idx):
        
        x, y, label =  batch["phrase"], batch["target"], batch["label"]
        y_hat = self.forward(x)
        loss = F.cosine_embedding_loss(y_hat, y, label)
        self.log('val_loss', loss)

        return loss

    def configure_optimizers(self):
        '''
        Configure optimizers for lightning module
        '''

        return AdamW(self.parameters(), lr=1e-5)
    
    # def 

if __name__=="__main__":

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("train_data", type=str)
    parser.add_argument("val_data", type=str)
    parser.add_argument("index_dir", type=str)
    parser.add_argument("n_epochs", type=int)

    args = parser.parse_args()
    train_dataset = data_loader.XLingualTrainDataset_baseline_lstm(args.train_data, args.index_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10, drop_last = True)

    val_dataset = data_loader.XLingualTrainDataset_baseline_lstm(args.val_data, args.index_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=10, drop_last = True)

    trainer = pl.Trainer(gpus=-1, max_epochs=args.n_epochs, distributed_backend='dp', prepare_data_per_node=False, num_nodes = 1, num_sanity_val_steps=0)
    
    
    
    EMBEDDING_DIM = train_dataset.embeddings.vocab.vectors[0].shape
    PAD_IDX = train_dataset.embeddings.vocab.stoi[train_dataset.vocabulary.pad_token]
    UNK_IDX = train_dataset.embeddings.vocab.stoi[train_dataset.vocabulary.unk_token]


    encoder = LSTM_model(vocab_size = len(train_dataset.embeddings.vocab.stoi), input_dim = EMBEDDING_DIM, hidden_size = (EMBEDDING_DIM), num_layers = 5, padding_idx = PAD_IDX)

    encoder.embedding.weight.data.copy_(train_dataset.embeddings.vocab.vectors)
    map_network = torch.nn.Identity() #torch.nn.Linear(encoder.config.hidden_size, encoder.config.hidden_size)


    encoder.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    encoder.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    
    model = XlingualDictionary_lstm(encoder, map_network)
    trainer.fit(model, train_dataloader, val_dataloader)
