import torch
from torch import nn
from transformers import BertModel, RobertaModel, RobertaConfig, AutoModelForSequenceClassification, BertConfig, \
    AutoConfig, GPT2Model
import torch.nn.functional as F
from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss


class bert_clf(nn.Module):
    def __init__(self, args):
        super(bert_clf, self).__init__()
        self.args = args

        self.dropout = nn.Dropout(0.5).to(args.device)
        self.relu = nn.ReLU().to(args.device)
        self.configuration = AutoConfig.from_pretrained(args.model, local_files_only=False)
        if args.method == "ours":
            self.configuration.hidden_dropout_prob = 0.0
            self.configuration.attention_probs_dropout_prob = 0.0
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=args.model,
                                              config=self.configuration).to(args.device)
        self.bert.pooler = None
        self.bert.to(args.device)

        self.out = nn.Linear(self.bert.config.hidden_size, args.num_labels).to(args.device)

        self.cross_entropy = torch.nn.CrossEntropyLoss(reduce=False)

    def forward(self, batch_data_list):
        torch.clear_autocast_cache()

        input_ids, token_type_ids, attention_mask, labels = self.batchifize(self.args, batch_data_list)
        last_hidden = self.bert(input_ids=input_ids.squeeze(), token_type_ids=token_type_ids.squeeze(),
                                attention_mask=attention_mask.squeeze())
        cls = last_hidden[0][:, 0, :].to(self.args.device)

        if self.args.method != "ours":
            cls = self.dropout(cls).to(self.args.device)
        logits = self.out(cls).to(self.args.device)
        # print(logits.size())
        loss = self.cross_entropy(logits, labels)

        # loss = torch.mean(loss)

        return cls.to(self.args.device), logits.to(self.args.device), labels.to(self.args.device), loss.to(
            self.args.device)

    def batchifize(self, args, dict_data_in_list):
        input_ids, attention_masks, token_type_ids, labels = None, None, None, None
        for data in dict_data_in_list:
            if input_ids is None and token_type_ids is None and attention_masks is None and labels is None:

                input_ids = torch.unsqueeze(data[args.INPUT_IDS], 0)
                token_type_ids = torch.unsqueeze(data[args.TOKEN_TYPE_IDS], 0)
                attention_masks = torch.unsqueeze(data[args.ATTENTION_MASK], 0)
                labels = torch.unsqueeze(data[args.LABEL], 0)
            else:
                input_ids = torch.cat((input_ids, torch.unsqueeze(data[args.INPUT_IDS], 0)), dim=0)
                token_type_ids = torch.cat((token_type_ids, torch.unsqueeze(data[args.TOKEN_TYPE_IDS], 0)), dim=0)
                attention_masks = torch.cat((attention_masks, torch.unsqueeze(data[args.ATTENTION_MASK], 0)), dim=0)
                labels = torch.cat((labels, torch.unsqueeze(data[args.LABEL], 0)), dim=0)

        return input_ids.to(args.device), token_type_ids.to(args.device), attention_masks.to(args.device), labels.to(
            args.device)


class gpt2_clf(nn.Module):
    def __init__(self, args):
        super(gpt2_clf, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(0.5).to(args.device)
        self.relu = nn.ReLU().to(args.device)
        self.configuration = AutoConfig.from_pretrained(args.model, local_files_only=True)
        if args.method == "ours":
            self.configuration.hidden_dropout_prob = 0.0
            self.configuration.attention_probs_dropout_prob = 0.0
        self.gpt2 = GPT2Model.from_pretrained(pretrained_model_name_or_path=args.model,
                                              config=self.configuration).to(args.device)
        self.gpt2.pooler = None
        self.gpt2.to(args.device)

        self.out = nn.Linear(self.gpt2.config.hidden_size, args.num_labels).to(args.device)

        self.cross_entropy = torch.nn.CrossEntropyLoss(reduce=False)

    def forward(self, batch_data_list):
        input_ids, token_type_ids, attention_mask, labels = self.batchifize(self.args, batch_data_list)
        last_hidden = self.gpt2(input_ids=input_ids.squeeze(), token_type_ids=token_type_ids.squeeze(),
                                attention_mask=attention_mask.squeeze())
        last_hidden = last_hidden.last_hidden_state
        cls = last_hidden[:, -1, :].to(self.args.device)
        if self.args.method != "ours":
            cls = self.dropout(cls).to(self.args.device)

        logits = self.out(cls).to(self.args.device)
        # print(logits.size())
        loss = self.cross_entropy(logits, labels)

        # loss = torch.mean(loss)

        return cls.to(self.args.device), logits.to(self.args.device), labels.to(self.args.device), loss.to(
            self.args.device)

    def batchifize(self, args, dict_data_in_list):
        input_ids, token_type_ids, attention_masks, labels = None, None, None, None
        for data in dict_data_in_list:
            if input_ids is None and attention_masks is None and labels is None:

                input_ids = torch.unsqueeze(data[args.INPUT_IDS], 0)
                token_type_ids = torch.unsqueeze(data[args.TOKEN_TYPE_IDS], 0)
                attention_masks = torch.unsqueeze(data[args.ATTENTION_MASK], 0)
                labels = torch.unsqueeze(data[args.LABEL], 0)
            else:
                input_ids = torch.cat((input_ids, torch.unsqueeze(data[args.INPUT_IDS], 0)), dim=0)
                token_type_ids = torch.cat((token_type_ids, torch.unsqueeze(data[args.TOKEN_TYPE_IDS], 0)), dim=0)
                attention_masks = torch.cat((attention_masks, torch.unsqueeze(data[args.ATTENTION_MASK], 0)), dim=0)
                labels = torch.cat((labels, torch.unsqueeze(data[args.LABEL], 0)), dim=0)

        return input_ids.to(args.device), token_type_ids.to(args.device), attention_masks.to(args.device), labels.to(
            args.device)

