from __future__ import division
import argparse
import transformers
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from torch.optim import AdamW
import random
from transformers import BertTokenizer, GPT2Tokenizer
from models import *
from utils import *

transformers.logging.set_verbosity_error()


def freeze_embedding(model):
    for n, p in model.named_parameters():
        if 'embed' in n or "wpe.weight" in n or 'wte.weight' in n:
            p.requires_grad = False


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_accuracy(preds, labels):
    correct = 0
    for p, l in zip(preds, labels):
        if p == l: correct += 1
    return correct / len(labels)


def get_f1(preds, labels):
    f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    return f1


def get_mcc(preds, labels):
    return matthews_corrcoef(y_true=labels, y_pred=preds)


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
    }


def get_test_performance(args, tokenizer, data_path, model):
    task2metric = {'CoLA': 'mcc', 'SST': 'acc', 'MNLI': 'acc', 'QNLI': 'acc',
                   'RTE': 'acc', 'QQP': 'f1', "weibo": 'f1'}

    metric = task2metric[args.task_name]

    test_dataset = bucket_dataset(args, tokenizer, data_path)

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=100, collate_fn=lambda x: x)

    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_dataloader:
            # input_ids, token_type_ids, attention_mask, labels = get_batch2(batch)

            # loss 1
            _, logits, labels, loss1 = model(batch)
            pred_softmax = torch.nn.functional.softmax(logits, dim=1)
            pred = np.argmax(pred_softmax.detach().cpu().numpy(), axis=1)
            preds.extend(pred)
            true_label = labels.detach().cpu().numpy()
            true_labels.extend(true_label)

    if metric == 'f1':
        return get_f1(preds, true_labels)
    elif metric == 'mcc':
        return get_mcc(preds, true_labels)
    elif metric == 'acc':
        return get_accuracy(preds, true_labels)


def pac_tuning(args):
    set_seed(args)
    if args.model == 'bert-base-uncased':
        model = bert_clf(args).to(args.device)
        tokenizer = BertTokenizer.from_pretrained(args.model, local_files_only=False)
    else:
        model = gpt2_clf(args).to(args.device)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model, local_files_only=False)
        tokenizer.pad_token = tokenizer.eos_token

    freeze_embedding(model)

    others = []
    min_gamma = 10
    max_gamma = 10
    prior_list, K_list = compute_K_sample_transformer(args, model, tokenizer, args.train_data, min_gamma, max_gamma)

    w0, p, layers, pretrain_dim, clf_dim, p_mean_pretrain, p_mean_clf = initialization(args, model)
    b = nn.Parameter(torch.zeros(layers, device=args.device), requires_grad=True)
    b.data[:layers - 2] += float(p_mean_pretrain)
    b.data[layers - 2:] += float(p_mean_clf)

    opt1 = AdamW(
        [{'params': [param for n, param in model.named_parameters() if param.requires_grad and args.model in n],
          'lr': args.lr4pretrain},
         {'params': [param for n, param in model.named_parameters() if
                     param.requires_grad and args.model not in n],
          'lr': args.lr4clf}], lr=args.lr4pretrain, weight_decay=args.weight_decay)
    opt2 = myAdam(args, pretrain_dim, [p], lr=0.5,
                  weight_decay=0.0)  # AdamW([p], lr=args.lr4pretrain, weight_decay=0)
    opt3 = myAdam(args, layers - 2, [b], lr=0.5, weight_decay=0.0)

    update = 0

    for epoch in range(args.max_epoch):
        torch.cuda.empty_cache()
        model.train()
        train_dataset = bucket_dataset(args, tokenizer, args.train_data)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=lambda x: x)
        for batch_idx, batch in enumerate(train_dataloader):
            opt1.zero_grad()
            opt2.zero_grad()
            opt3.zero_grad()
            # noise injection and ||w-w0||^2
            wdecay = weight_decay_mulb(model, b, w0)  # weight_decay(model_copy, w0)  # weight_decay_mulb(model, b, w0)
            """noise injection"""
            noises, noises_scaled = noise_injection(model, p)
            """loss1 is the cross-entropy loss"""
            _, _, _, loss1 = model(batch)

            if epoch < args.stage1_epochs:
                kl = get_kl_term_layer_pb(model, wdecay, p, b)
                K = fun_K_auto(torch.exp(b.mean()), prior_list, K_list)
                gamma1_ = K ** (-1) * (2 * (kl + 10 + layers * 3) / args.train_size / 3) ** 0.5
                gamma1 = torch.clip(gamma1_, max=max_gamma, min=min_gamma)
                """loss2 the PAC loss"""
                loss2 = 3 * K ** 2 * gamma1 / 2 + (kl + 10 + layers * 3) / args.train_size / gamma1
            else:
                """after stage 1, we have the standard PGD"""
                loss2 = 0 * loss1

            loss1.mean().backward(retain_graph=True)
            """update the parameters of noise in the stage 1 only"""
            if epoch < args.stage1_epochs:
                kl_term_backward(loss2, model, p, noises)
            """remove noise after computing gradient"""
            rm_injected_noises(model, noises_scaled)

            opt1.step()
            update += 1

            if epoch < args.stage1_epochs:
                opt2.step()
                opt3.step()

            others.append([p.mean().cpu().item()])

        valid_performance = get_test_performance(args, tokenizer, args.valid_data, model)

        print("task:{}\tseed:{}\tepoch:{}\tvalid_performance:{}\tmodel:{}".format(
                    args.task_name, args.seed, epoch,
                    valid_performance, args.model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_name",
        default="SST",
        type=str,
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    parser.add_argument("--lr4pretrain",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--lr4clf",
                        default=1e-2,  # 1e-2
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay",
                        default=1e-2,
                        type=float,
                        help="Weight delay if we apply some.")

    parser.add_argument("--max_epoch",
                        default=250,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--attention_dropout", default=0.5, type=float)

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument("--INPUT_IDS", type=str, default='input_ids')
    parser.add_argument("--ATTENTION_MASK", type=str, default='attention_mask')
    parser.add_argument("--TOKEN_TYPE_IDS", type=str, default="token_type_ids")
    parser.add_argument("--LABEL", type=str, default='label')
    parser.add_argument('--TEXT', type=str, default='text')
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", default=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    parser.add_argument("--train_data", type=str, default="data/SST/train.txt")
    parser.add_argument("--valid_data", type=str, default="data/SST/dev.txt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument('--representation_dim', type=int, default=768)
    parser.add_argument("--stage1_epochs", default=200, type=int, help="The number of epochs for stage 1.")
    parser.add_argument('--model', default='bert-base-uncased', type=str)  # gpt2

    args = parser.parse_args()

    pac_tuning(args)
