#!/bin/bash
#SBATCH -A research
#SBATCH -n 20
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=saujas.vaduguru@research.iiit.ac.in

mkdir ../../models/st_w_linear_results_ALL ../../models/st_w_linear_results_EN ../../models/st_w_linear_results_HI ../../models/st_w_linear_results_BE ../../models/st_w_linear_results_GU ../../models/st_w_linear_results_OD ../../models/st_w_linear_results_PU ../../models/st_w_linear_results_MA

python sentence_transformer.py ../../data/filtered/train.json ../../data/filtered/validation.json ../../data/filtered/test.json ../../models/smallidx/ ../../models/st_w_linear_results_ALL ../../models/st_w_linear_ALL.pt 5
python sentence_transformer.py ../../data/filtered/train.json ../../data/filtered/validation.json ../../data/filtered/test.json ../../models/smallidx/ ../../models/st_w_linear_results_EN ../../models/st_w_linear_EN.pt 5 --langs EN
python sentence_transformer.py ../../data/filtered/train.json ../../data/filtered/validation.json ../../data/filtered/test.json ../../models/smallidx/ ../../models/st_w_linear_results_HI ../../models/st_w_linear_HI.pt 5 --langs HI
python sentence_transformer.py ../../data/filtered/train.json ../../data/filtered/validation.json ../../data/filtered/test.json ../../models/smallidx/ ../../models/st_w_linear_results_BE ../../models/st_w_linear_BE.pt 5 --langs BE
python sentence_transformer.py ../../data/filtered/train.json ../../data/filtered/validation.json ../../data/filtered/test.json ../../models/smallidx/ ../../models/st_w_linear_results_GU ../../models/st_w_linear_GU.pt 5 --langs GU
python sentence_transformer.py ../../data/filtered/train.json ../../data/filtered/validation.json ../../data/filtered/test.json ../../models/smallidx/ ../../models/st_w_linear_results_OD ../../models/st_w_linear_OD.pt 5 --langs OD
python sentence_transformer.py ../../data/filtered/train.json ../../data/filtered/validation.json ../../data/filtered/test.json ../../models/smallidx/ ../../models/st_w_linear_results_PU ../../models/st_w_linear_PU.pt 5 --langs PU
python sentence_transformer.py ../../data/filtered/train.json ../../data/filtered/validation.json ../../data/filtered/test.json ../../models/smallidx/ ../../models/st_w_linear_results_MA ../../models/st_w_linear_MA.pt 5 --langs MA
