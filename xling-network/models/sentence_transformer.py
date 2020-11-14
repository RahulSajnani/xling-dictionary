from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, models, evaluation
from torch.utils.data import DataLoader
import json
import argparse
from collections import defaultdict
import os

def get_IR_data(data_path, index_dir):
    with open(data_path, 'r') as f:
        data = json.load(f)

    data_by_lang = defaultdict(list)
    for d in data:
        data_by_lang[d["Target_ID"]].append(d)
    
    queries, corpus, relevant_docs = dict(), dict(), dict()
    for lang, dataset in data_by_lang.items():
        lang_corpus = dict()
        lang_reverse_corpus = dict()
        i = 0
        with open(os.path.join(index_dir, "{}.vocab".format(lang)), 'r') as f:
            for line in f:
                corpus[str(i)] = line.strip()
                lang_reverse_corpus[line.strip()] = str(i)
                i += 1
        
        lang_queries = dict()
        lang_relevant_docs = dict()
        i = 0
        for d in dataset:
            lang_queries[str(i)] = d["Source_text"]
            lang_relevant_docs[str(i)] = {lang_reverse_corpus[d["Target_keyword"]]}
            i += 1
        
        queries[lang] = lang_queries
        corpus[lang] = lang_corpus
        relevant_docs[lang] = lang_relevant_docs
    
    return queries, corpus, relevant_docs

if __name__=="__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("train_data", type=str)
    parser.add_argument("val_data", type=str)
    parser.add_argument("test_data", type=str)
    parser.add_argument("index_dir", type=str)
    parser.add_argument("results_dir", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("n_epochs", type=int)
    parser.add_argument("k", type=int)

    args = parser.parse_args()

    word_embedding_model = models.Transformer('ai4bharat/indic-bert', max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    with open(args.train_data, 'r') as f:
        data = json.load(f)
    train_examples = [InputExample(texts=[d["Source_text"], d["Target_keyword"]]) for d in data]

    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    train_loss = losses.MSELoss(model)

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)
    model.save(args.model_path)

    queries, corpus, relevant_docs = get_IR_data(args.val_data, args.index_dir)

    for lang in queries:
        evaluator = evaluation.InformationRetrievalEvaluator(queries[lang], 
                                                                corpus[lang], 
                                                                relevant_docs[lang],
                                                                mrr_at_k=[args.k],
                                                                ndcg_at_k=[args.k],
                                                                accuracy_at_k=[args.k],
                                                                precision_recall_at_k=[args.k],
                                                                map_at_k=[args.k])
        evaluator(model, output_path=os.path.join(args.results_dir, "{}.results".format(lang)))