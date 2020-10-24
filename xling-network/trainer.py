from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import data_loader 




class Xlingual_Trainer(pl.LightningModule):
    
    '''
    Cross lingual dictionary trainer
    '''

    def __init__(self, indic_bert, learning_rate=1e-3, dataset_path = "../data/temp_data.json"):
        
        super().__init__()
        self.save_hyperparameters()    
        self.xlingual_bert = indic_bert

    def forward(self, x):
        
        embedding = self.xlingual_bert(x)
        return embedding

    def training_step(self, batch, batch_idx):
        
        x = batch
        y_hat = self.xlingual_bert(x["phrase"])
        # loss = F.cross_entropy(y_hat, y)
        # self.log('train_loss', loss, on_epoch=True)
        # return loss

    def validation_step(self, batch, batch_idx):
        
        x = batch
        y_hat = self.xlingual_bert(x["phrase"])
        # loss = F.cross_entropy(y_hat, y)
        # self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        
        x = batch
        y_hat = self.xlingual_bert(x["phrase"])
        
        # self.log('test_loss', loss)

    def configure_optimizers(self):
        '''
        Configure optimizers for lightning module
        '''
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self) -> DataLoader:
        '''
        Configure train dataloader
        '''
        dataset = data_loader.XLingual_loader(dataset_path = dataset_path)
        data_loader = DataLoader(dataset, batch_size = 32, shuffle=True)


    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

if __name__=="__main__":
    
    pass