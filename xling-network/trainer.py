import argparse

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import data_loader
import faiss
from transformers import AdamW, AutoModel
from model import XlingualDictionaryBERT

if __name__=="__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("train_data", type=str)
    parser.add_argument("val_data", type=str)
    parser.add_argument("test_data", type=str)
    parser.add_argument("index_dir", type=str)
    parser.add_argument("encoder_cache_dir", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("n_epochs", type=int)
    parser.add_argument("k", type=int)

    args = parser.parse_args()
    train_dataset = data_loader.XLingualDataset(args.train_data, args.index_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=10, drop_last=True, collate_fn=data_loader.get_train_collate())

    val_dataset = data_loader.XLingualDataset(args.val_data, args.index_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=10, drop_last=True, collate_fn=data_loader.get_eval_collate(args.index_dir, args.k))

    test_dataset = data_loader.XLingualDataset(args.test_data, args.index_dir)
    test_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=10, drop_last=True, collate_fn=data_loader.get_eval_collate(args.index_dir, args.k))

    trainer = pl.Trainer(gpus=-1, max_epochs=args.n_epochs, distributed_backend='dp', prepare_data_per_node=False, num_nodes=1) #, progress_bar_refresh_rate=0)

    model = XlingualDictionaryBERT()
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.save_checkpoint(args.model_path)

    # trainer.test(test_dataloaders=test_dataloader)
