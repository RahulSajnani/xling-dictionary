import argparse

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import data_loader, loader_test
import faiss
import os,sys
from trainer import XlingualDictionary
from collections import defaultdict
from transformers import AutoTokenizer
sys.path.append("../")
from models.lstm import LSTM_model
from helper_functions import read_json_file


def preprocess_text(phrase, src_lang, target_word, dataset):
    '''

    preprocess text
    '''

    tokens = dataset.preprocessing_data(phrase, src_lang, src=True)
    input_idx = dataset.tokens2tensor(tokens)

    #target = torch.tensor(self.targets[idx])
    #target = (target_word)
    label = torch.ones(input_idx.shape[0], 1)
    return {
            "phrase": {
                        'input_ids': input_idx.squeeze(),
                    },
            #"target": target,
            "label": label
        }

def get_accuracy(test_data, model, index_dir, test_dataset, k=10, batch_size=32):

    model.eval()

    lang_map = {'HI': 'hi', 'BE': 'bn', 'GU': 'gu', 'OD': 'or', 'PU': 'pa', 'EN': 'en', 'MA': 'mr'}
    data_by_lang = defaultdict(list)

    for d in test_data:
        data_by_lang[d["Target_ID"]].append(d)

    correct = 0
    total = 0

    for lang in lang_map.keys():
        with open(os.path.join(index_dir, lang_map[lang] + ".vocab"), 'r') as f:
            vocab = [line.strip() for line in f]

        phrases = [d["Source_text"] for d in data_by_lang[lang]]
        targets = [d["Target_keyword"] for d in data_by_lang[lang]]
        b = 0

        with torch.no_grad():

            for i in range(len(phrases)):

                phrase = phrases[i]
                src_lang = lang_map[lang]
                target_word = targets[i]

                batch = preprocess_text(phrase, src_lang, target_word, test_dataset)

                x, label =  batch["phrase"]["input_ids"].unsqueeze(0).cuda(),  batch["label"].unsqueeze(0).cuda()
                y_hat = model(x)
                #y_hat = y_hat.squeeze()
                #print(y_hat.shape)
                index = faiss.read_index(os.path.join(index_dir, lang_map[lang] + ".index"))
                # embeddings = torch.mean(outputs.last_hidden_state, 1).detach().cpu().numpy()
                y_hat = y_hat.detach().cpu().numpy()
                faiss.normalize_L2(y_hat)
                D, I = index.search(y_hat, k)

                words = [[vocab[i] for i in row] for row in I]

                if target_word in words[0]:
                    correct+=1

                total += 1
                print(correct/total)
                if total % ( batch_size) == 0:
                    print("{} samples processed {}".format(b, correct/total))

        print("{} done!".format(lang))

    return correct / total


def xlingual_reverse_dictionary(model, index_dir, k):

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")

    #lang_map = {"EN": "en"}
    lang_map = {'HI': 'hi', 'BE': 'bn', 'GU': 'gu', 'OD': 'or', 'PU': 'pa', 'EN': 'en', 'MA': 'mr'}
    data_by_lang = defaultdict(list)
    vocab = {}


    for lang in lang_map.keys():
        with open(os.path.join(index_dir, lang_map[lang] + ".vocab"), 'r') as f:
            vocab[lang_map[lang]] = [line.strip() for line in f]


    lang = input("Enter output language from HI-BE-GU-OD-PU-EN-MA \n")

    with torch.no_grad():
        while (1):
            phrases = input("Enter source phrase:\n")
            phrases = [phrases.lower()]

            print(phrases)
            tokens = tokenizer(phrases, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            tokens.to('cuda')
            _, I = model(tokens, os.path.join(index_dir, lang_map[lang] + ".index"), k)
            print(I)
            words = [[vocab[lang_map[lang]][i] for i in row] for row in I]
            for word in words[0]:
                print(word)
            # print(words)
            #print(words)

if __name__ == "__main__":


    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("test_data", type=str)
    parser.add_argument("index_dir", type=str)
    parser.add_argument("emb_path", type=str)
    parser.add_argument("emb_dir", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("k", type=int, default=10)
    args = parser.parse_args()



    test_dataset = loader_test.XLingualTrainDataset_baseline_lstm(args.test_data, args.emb_path, args.emb_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=10, drop_last = True)

    EMBEDDING_DIM = test_dataset.embeddings.vectors[0].shape[0]
    PAD_IDX = test_dataset.embeddings.stoi[test_dataset.vocabulary.pad_token]
    UNK_IDX = test_dataset.embeddings.stoi[test_dataset.vocabulary.unk_token]

    model = LSTM_model(vocab_size = len(test_dataset.embeddings), input_dim = EMBEDDING_DIM, hidden_size = (EMBEDDING_DIM), num_layers = 5, padding_idx = PAD_IDX)
    model.embedding.weight.data.copy_(test_dataset.embeddings.vectors)
    model.load_state_dict(torch.load(args.model_path))
    model.to('cuda')

    # if args.test_data == "test":
    #     xlingual_reverse_dictionary(model, index_dir=args.index_dir, k = args.k)
    # elif os.path.exists(args.test_data):
    print(get_accuracy(read_json_file(args.test_data), model, args.index_dir, test_dataset, args.k))
