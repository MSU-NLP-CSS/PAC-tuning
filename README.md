# PAC-tuning

This repo is for the paper PAC-tuning: Fine-tuning Pretrained Language Models with PAC-driven Perturbed Gradient Descent to appear at EMNLP 23. In this paper, we propose a two stage fine-tuning method, namely PAC-tuning, that learns noise levels for Perturbed Gradient Descent, by directly minimizing a PAC-Bayes upperbound. PAC-tuning alleviates the requirements for heavy hyperparameter search and proves that PAC-Bayes training can be utilized to a challenging scenario of extremely large model parameters and small data size.

## Environment:
* PyTorch  
* transformers

## To run PAC-tuning:
```
python pac_tuning.py  --num_labels 2 
                      --task_name SST 
                      --train_data data/SST/train.txt  
                      --valid_data data/SST/dev.txt 
                      --batch_size 100 
                      --train_size 100 
                      --max_epoch 250 
                      --stage1_epochs 200
                      --model bert-base-uncased
```

### Note
It is better to have a larger batch size for PAC-tuning a Pretrained Languge Model on downstream text classification tasks. Please check the Section **6** in our paper for suggestions about how to use PAC-tuning for your specific tasks.