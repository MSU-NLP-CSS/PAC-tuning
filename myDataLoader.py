from __future__ import division

import os

os.environ['TRANSFORMERS_CACHE'] = '/scratch0/liuguan5/pretrained_models/'
import random
from torch.utils.data import IterableDataset, Dataset
import transformers
import numpy as np
import torch

torch.cuda.empty_cache()
transformers.logging.set_verbosity_error()
#bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
#gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
#gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token


class bucket_dataset(Dataset):
    def __init__(self, args, tokenizer, source_data_file, target_data_file=None, need_shuffle=False, size=None):
        super(bucket_dataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        if need_shuffle:
            self.__random__(source_data_file, target_data_file)
        else:
            if source_data_file != target_data_file:
                os.system("cat {0} > {1}".format(source_data_file, target_data_file))
        new_file = target_data_file
        if new_file is None:
            new_file = source_data_file
        self.data = open(new_file, "r").readlines()
        if size is not None:
            self.data = self.data[:min(size, len(self.data))]

    def __random__(self, source_data_file, target_data_file):
        data_list = []
        for line in open(source_data_file, 'r'):
            data_list.append(line.strip())
        random.shuffle(data_list)
        with open(target_data_file, 'w') as writer:
            writer.write('\n'.join(data_list))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        if self.args.pair_input == "True":
            encodings = convert_pair_sentence_to_ids(self.args, self.data[idx])
        else:
            encodings = convert_single_sentence_to_ids(self.args, self.data[idx])
        '''

        return text2ids(self.args,self.tokenizer,self.data[idx])


def text2ids(args, tokenizer, input):
    encodings = {
        args.INPUT_IDS: None,
        args.ATTENTION_MASK: None,
        args.TOKEN_TYPE_IDS: None
    }

    split_input = input.strip().split("\t")
    label = int(split_input[-1])
    if len(split_input) > 2:
        target, claim, label = split_input[0], split_input[1], int(split_input[2])

        encoded_dict = tokenizer.encode_plus(
            target, claim,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_length,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_token_type_ids=True,
            return_tensors='pt',
            truncation=True,
        )
    else:
        text, label = split_input[0], int(split_input[1])
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_length,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_token_type_ids=True,
            return_tensors='pt',
            truncation=True,

        )

    encodings[args.INPUT_IDS] = torch.from_numpy(np.array(encoded_dict['input_ids'])).to(args.device)
    encodings[args.TOKEN_TYPE_IDS] = torch.from_numpy(np.array(encoded_dict['token_type_ids'])).to(args.device)
    encodings[args.ATTENTION_MASK] = torch.from_numpy(np.array(encoded_dict['attention_mask'])).to(args.device)
    encodings[args.LABEL] = torch.from_numpy(np.array(label)).to(args.device)

    return encodings


def batchifize(args, dict_data_in_list):
    input_ids, attention_masks, token_type_ids, labels = None, None, None, None
    for data in dict_data_in_list:
        if input_ids is None and token_type_ids is None and attention_masks is None and labels is None:

            input_ids = torch.unsqueeze(data[args.INPUT_IDS], 0)
            # token_type_ids = torch.unsqueeze(data[args.TOKEN_TYPE_IDS], 0)
            attention_masks = torch.unsqueeze(data[args.ATTENTION_MASK], 0)
            labels = torch.unsqueeze(data[args.LABEL], 0)
        else:
            input_ids = torch.cat((input_ids, torch.unsqueeze(data[args.INPUT_IDS], 0)), dim=0)
            # token_type_ids = torch.cat((token_type_ids, torch.unsqueeze(data[args.TOKEN_TYPE_IDS], 0)), dim=0)
            attention_masks = torch.cat((attention_masks, torch.unsqueeze(data[args.ATTENTION_MASK], 0)), dim=0)
            labels = torch.cat((labels, torch.unsqueeze(data[args.LABEL], 0)), dim=0)

    return input_ids.to(args.device), attention_masks.to(args.device), labels.to(args.device)


def split_train_valid(args):
    data_list = []

    for line in open(args._train, 'r'):
        data_list.append(line.strip())

    random.shuffle(data_list)

    train = data_list[:int(1 - args.train_size * args.valid_train_ratio)]
    valid = data_list[int(1 - args.train_size * args.valid_train_ratio):]

    with open(args.train_, 'w') as writer:
        writer.write('\n'.join(train))

    with open(args.valid_, 'w') as writer:
        writer.write("\n".join(valid))

    return len(train), len(valid)


INPUT_IDS, TOKEN_TYPE_IDS, ATTENTION_MASK, LABEL = 'input_ids', 'token_type_ids', 'attention_mask', 'label'


def get_batch2(args, batch):
    input_ids, token_type_ids, attention_mask, labels = None, None, None, None
    for data in batch:
        if input_ids is None:
            # print(type(data),data)
            input_ids = torch.unsqueeze(data[INPUT_IDS], 0)
            token_type_ids = torch.unsqueeze(data[TOKEN_TYPE_IDS], 0)
            attention_mask = torch.unsqueeze(data[ATTENTION_MASK], 0)
            labels = torch.unsqueeze(data[LABEL], 0)
        else:
            input_ids = torch.cat((input_ids, torch.unsqueeze(data[INPUT_IDS], 0)), 0)
            # print(data[INPUT_IDS].shape)
            token_type_ids = torch.cat((token_type_ids, torch.unsqueeze(data[TOKEN_TYPE_IDS], 0)), 0)
            attention_mask = torch.cat((attention_mask, torch.unsqueeze(data[ATTENTION_MASK], 0)), 0)
            # print(labels.shape,data[LABEL].shape)
            labels = torch.cat((labels, torch.unsqueeze(data[LABEL], 0)))

    return input_ids.to(args.device), token_type_ids.to(args.device), attention_mask.to(args.device), labels.to(
        args.device)
