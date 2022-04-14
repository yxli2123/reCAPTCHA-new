import torch
import numpy as np
import random
import torch.nn.functional as F
import json
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def prob(w1, w2):
    w1_token = tokenizer(w1, add_special_tokens=False, return_tensors="pt")['input_ids']
    w2_token = tokenizer(w2, add_special_tokens=False, return_tensors="pt")['input_ids']
    # print(w1_token, w2_token)

    mask = tokenizer.mask_token
    w1_given_w2 = tokenizer.encode_plus(mask + w2, return_tensors="pt")
    w2_given_w1 = tokenizer.encode_plus(w1 + mask, return_tensors="pt")

    mask_index_w1_given_w2 = torch.where(w1_given_w2["input_ids"][0] == tokenizer.mask_token_id)
    mask_index_w2_given_w1 = torch.where(w2_given_w1["input_ids"][0] == tokenizer.mask_token_id)

    logits_w1_given_w2 = model(**w1_given_w2).logits[0, mask_index_w1_given_w2]
    logits_w2_given_w1 = model(**w2_given_w1).logits[0, mask_index_w2_given_w1]

    p_w1_given_w2 = F.softmax(logits_w1_given_w2, dim=-1)
    p_w2_given_w1 = F.softmax(logits_w2_given_w1, dim=-1)
    # print(p_w1_given_w2.shape)

    # val_w1_given_w2, pos_w1_given_w2 = torch.topk(p_w1_given_w2, k=5)
    # val_w2_given_w1, pos_w2_given_w1 = torch.topk(p_w2_given_w1, k=5)

    # print(val_w1_given_w2, val_w2_given_w1)
    # print(tokenizer.decode(pos_w1_given_w2[0]), tokenizer.decode(pos_w2_given_w1[0]))

    p_w1_given_w2 = p_w1_given_w2[0, w1_token]
    p_w2_given_w1 = p_w2_given_w1[0, w2_token]
    # print(p_w1_given_w2, p_w2_given_w1)

    return p_w1_given_w2, p_w2_given_w1


def evaluate_segmentation(y_gt, y_pr):
    y_gt, y_pr = y_gt.flatten(), y_pr.flatten()

    matrix = confusion_matrix(y_gt, y_pr)
    acc = accuracy_score(y_gt, y_pr)
    f1 = f1_score(y_gt, y_pr)

    metric = {"confusion_matrix": matrix,
              "accuracy": acc,
              "f1_score": f1}

    return metric


def evaluate(y_gt, y_pr, num_character=1, top_k=1, decode_file='../data/text/decode.json'):
    """
    :param          y_gt: of shape (N*num_char)
    :param          y_pr: of shape (N*num_char, num_cls)
    :param num_character: 1 or 2
    :param         top_k: top k candidates of the prediction
    :param   decode_file: decode
    :return:
    """

    # Statics
    acc_single = 0
    acc_pair = 0
    acc_topk = 0
    loss = F.cross_entropy(y_pr, y_gt).item()

    # Results
    with open(decode_file, 'r') as fp:
        decode_dict = json.load(fp)

    def decode(token):
        """
        :param token: of shape (N, ), int type
        :return: a string
        """
        words = []
        for t in token:
            words.append(decode_dict[f'{t.item()}'])
        words = "".join(words)
        return words

    results = []

    N_num_char, num_cls = y_pr.shape
    y_pr = y_pr.view(N_num_char // num_character, num_character, num_cls)   # (N, num_char, num_cls)
    y_gt = y_gt.view(N_num_char // num_character, num_character)            # (N, num_char)
    N, _, _ = y_pr.shape

    for gt, pr in zip(y_gt, y_pr):
        """
        pr of shape (num_char, num_cls)
        gt of shape (num_char)
        """
        pr = F.softmax(pr, dim=1)
        top_k_value, top_k_token = torch.topk(pr, k=top_k, dim=1)  # (num_char, k)

        # Statics
        correct_cnt = 0
        char_pr = []
        for i in range(num_character):
            # Single character accuracy
            if top_k_token[i][0] == gt[i]:
                correct_cnt += 1
                acc_single += 1

            # Top k character accuracy: if ground truth is among the top k answer
            for j in range(top_k):
                if top_k_token[i][j] == gt[i]:
                    acc_topk += 1

            if correct_cnt == num_character:
                acc_pair += 1
            char_pr_i = {decode([token]): pr[i, token].item() for token in top_k_token[i]}
            char_pr.append(char_pr_i)

        # Results
        char_gt = decode(gt)   # e.g. "中国"
        results.append({"char_gt": char_gt,
                        "char_pr": char_pr})

    metric = {"loss": loss,
              "acc_single": acc_single / (num_character * N),
              "acc_pair": acc_pair / N,
              "acc_topk": acc_topk / (num_character * N),
              "results": results}
    return metric


def evaluate_with_LM(y_gt, y_pr, num_character=1, top_k=1,
                     decode_file='../data/text/decode.json',
                     LM_file='../data/text/P_condition.npy'):
    """
    :param          y_gt: of shape (N*num_char)
    :param          y_pr: of shape (N*num_char, num_cls)
    :param num_character: 1 or 2
    :param         top_k: top k candidates of the prediction
    :param   decode_file: decode
    :param       LM_file: path to the 2-gram language model
    :return:
    """

    # Load LM
    P_LM = np.load(LM_file)  # (200, 200, 2)
    P_LM = torch.tensor(P_LM, device=y_gt.device)

    # Statics
    acc_single = 0
    acc_pair = 0
    acc_topk = 0
    loss = F.cross_entropy(y_pr, y_gt)

    # Results
    with open(decode_file, 'r') as fp:
        decode_dict = json.load(fp)

    def decode(token):
        """
        :param token: of shape (N, ), int type
        :return: a string
        """
        words = []
        for t in token:
            words.append(decode_dict[t])
        words = "".join(words)
        return words

    results = []

    N_num_char, num_cls = y_pr.shape
    y_pr = y_pr.view(N_num_char // num_character, num_character, num_cls)   # (N, num_char, num_cls)
    y_gt = y_gt.view(N_num_char // num_character, num_character)            # (N, num_char)
    N, _, _ = y_pr.shape

    for gt, pr in zip(y_gt, y_pr):
        """
        pr of shape (num_char, num_cls)
        gt of shape (num_char)
        """

        pr = F.softmax(pr, dim=1)
        pr = 0.5 * (pr*P_LM)
        top_k_value, top_k_token = torch.topk(pr, k=top_k, dim=1)

        # Statics
        correct_cnt = 0
        for i in range(num_character):
            # Single character accuracy
            if top_k_token[i][0] == gt[i]:
                correct_cnt += 1
                acc_single += 1

            # Top k character accuracy: if ground truth is among the top k answer
            for j in range(top_k):
                if top_k_token[i][j] == gt[i]:
                    acc_topk += 1

            if correct_cnt == num_character:
                acc_pair += 1

        # Results

        char_gt = decode(gt)   # e.g. "中国"
        char_pr = " ".join([decode(tokens) for tokens in top_k_token])  # e.g. "中口申 国图固"
        results.append({"char_gt": char_gt,
                        "char_pr": char_pr})

    metric = {"loss": loss,
              "acc_single": acc_single / (num_character * N),
              "acc_pair": acc_pair / N,
              "acc_topk": acc_topk / (num_character * N),
              "results": results}
    return metric

