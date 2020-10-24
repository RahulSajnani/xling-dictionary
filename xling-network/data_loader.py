import json
import torch
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
from helper_functions import *
from transformers import AutoTokenizer

class XLingual_loader(Dataset):

    '''
    Reverse dictionary data loader
    '''

    def __init__(self, dataset_path):
        '''
        Init class method

        Arguments:
            dataset_path - path to json data
        

        '''

        print("Loading dataset......")
        self.dataset_json = read_json_file(dataset_path)
        print("Dataset loaded !!")

        print(len(self.dataset_json))
        # Tokenizer for Indian languages
        self.tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert", max_seq_length = 128, min_seq_length = 128)
        self.language_ids = {'HI': 0, 'BE': 1, 'GU': 2, 'OD': 3, 'PU': 4, 'EN': 5, 'MA': 6}


    def __getitem__(self, idx):
        '''
        Get item function pytorch
        
        Arguments:
            idx - text index 
        '''
        print(self.dataset_json[idx])

        data = self.convert_dict_2_features(self.dataset_json[idx])
        
        return self.input_to_tensor(data) 

    def convert_dict_2_features(self, text_dict):
        '''
        Function to convert input dictionary to features for training bert :)
        
        Arguments:
            text_dict - input dictionary

        Returns:
            data - encoded input features after tokenization
        '''

        data = {}
        
        data["phrase"]    = text_dict["Source_text"]
        data["target"]    = text_dict["Target_keyword"]
        data["src_id"]    = self.language_ids[text_dict["Source_ID"]]
        data["target_id"] = self.language_ids[text_dict["Target_ID"]]

        return data



    def __len__(self):

        '''
        Returns length of dataset
        '''
        
        return len(self.dataset_json)


    def input_to_tensor(self, data):
        '''
        Convert inputs to tensor
        '''
        out_dictionary = {}
            
        out_dictionary["input"]  = self.tokenizer(data["phrase"], padding = True)
        out_dictionary["target"] = self.tokenizer(data["target"], padding = True)

        # Language ids
        out_dictionary["src_id"]    = data["src_id"]
        out_dictionary["target_id"] = data["target_id"]

        print("check")
        # for keyword in out_dictionary:
        #     print("hello")
        #     if isinstance(out_dictionary[keyword], dict):
        #         print("lol")

        return out_dictionary   


   
if __name__ == "__main__":

    dataset_path = "../xling-dictionary/data/temp_data.json"

    dataset = XLingual_loader(dataset_path = dataset_path)
    print(dataset[0])
    # data_loader = DataLoader(dataset, batch_size = 2, shuffle=True)

    # for batch, data in enumerate(data_loader):

    #     for key in data:

    #         print(key, data[key])