from __future__ import division
import os
import random
from torch.utils.data import Dataset
import numpy as np
import torch

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
        return text2ids(self.args, self.tokenizer, self.data[idx])


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
