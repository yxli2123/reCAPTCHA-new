import torch
import numpy as np
import random
import torch.nn.functional as F
import json
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.cluster import KMeans


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def crop_image(image: np.ndarray, n_clusters=4, box_size=60):
    k_means_class = KMeans(n_clusters=n_clusters, init='k-means++')
    sample = np.where(image[..., 0] != 0)
    sample = np.array([[sample[0][i], sample[1][i]] for i in range(len(sample[0]))])
    kmeans = k_means_class.fit(sample)
    center = kmeans.cluster_centers_.astype('int64')

    image_stack = []
    H, W, C = image.shape
    for c in center:
        h, w = c[0], c[1]
        top = max(h - box_size // 2, 0)
        bottom = min(h + box_size // 2, H)
        left = max(w - box_size // 2, 0)
        right = min(w + box_size // 2, W)

        image_piece = image[top: bottom, left: right]
        image_stack.append(image_piece)

    image_stack = np.stack(image_stack)
    return image_stack


def evaluate_segmentation(y_gt, y_pr):
    y_gt, y_pr = y_gt.flatten(), y_pr.flatten()
    y_gt, y_pr = y_gt.cpu().numpy(), y_pr.cpu().numpy()

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

