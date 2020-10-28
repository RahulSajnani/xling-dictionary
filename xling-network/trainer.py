import argparse

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import data_loader
import faiss
from transformers import AdamW, AutoModel

class XlingualDictionary(pl.LightningModule):
    '''
    Cross lingual dictionary model
    '''

    def __init__(self, xling_encoder, map_network):
        
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
        outputs = self.encoder(**x)
        sequence_outputs = outputs[0]
        sequence_embedding = torch.mean(sequence_outputs, 1)
        y_hat = self.map(sequence_embedding)
        loss = F.mse_loss(y_hat, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y =  batch["phrase"], batch["target"]
        outputs = self.encoder(**x)
        sequence_outputs = outputs[0]
        sequence_embedding = torch.mean(sequence_outputs, 1)
        y_hat = self.map(sequence_embedding)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)


    def configure_optimizers(self):
        '''
        Configure optimizers for lightning module
        '''

        return AdamW(self.parameters(), lr=1e-5)

if __name__=="__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("train_data", type=str)
    parser.add_argument("val_data", type=str)
    parser.add_argument("index_dir", type=str)
    parser.add_argument("encoder_cache_dir", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("n_epochs", type=int)

    args = parser.parse_args()

    train_dataset = data_loader.XLingualTrainDataset(args.train_data, args.index_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10)

    val_dataset = data_loader.XLingualTrainDataset(args.val_data, args.index_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=10)

    trainer = pl.Trainer(gpus=1, max_epochs=args.n_epochs)
    encoder = AutoModel.from_pretrained("ai4bharat/indic-bert", cache_dir=args.encoder_cache_dir, return_dict=True)
    map_network = torch.nn.Linear(encoder.config.hidden_size, encoder.config.hidden_size)
    model = XlingualDictionary(encoder, map_network)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.save_checkpoint(args.model_path)
