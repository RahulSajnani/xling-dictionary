from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import data_loader
import faiss

class XlingualDictionary(pl.LightningModule):
    '''
    Cross lingual dictionary model
    '''

    def __init__(self, xling_encoder, map_network, learning_rate=1e-3, dataset_path="../data/temp_data.json"):
        
        super().__init__()
        
        self.save_hyperparameters()    
        self.encoder = xling_encoder
        self.map = map_network

    def forward(self, x, target_lang, index_paths):
        
        outputs = self.encoder(x)
        sequence_outputs = outputs[0]
        sequence_embedding = torch.mean(sequence_outputs, 1)
        y_hat = self.map(sequence_embedding)
        index = faiss.read_index(index_paths[target_lang])
        D, I = index.search(y_hat, 1)
        return y_hat, I

    def training_step(self, batch, batch_idx):


        x, y =  batch["phrase"], batch["target"]
        y_hat, I = self.forward(x)
        loss = F.mse_loss(y_hat, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        
        x, y =  batch["phrase"], batch["target"]
        y_hat, I = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)


    def configure_optimizers(self):
        '''
        Configure optimizers for lightning module
        '''
        return torch.optim.Adam(self.map.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self) -> DataLoader:
        '''
        Configure train dataloader
        '''
        dataset = data_loader.XLingualLoader(dataset_path=dataset_path)
        data_loader = DataLoader(dataset, batch_size = 32, shuffle=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

if __name__=="__main__":
    pass