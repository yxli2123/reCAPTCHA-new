import torch
import numpy as np
import random
import json
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.cluster import KMeans
from tqdm import tqdm
import einops


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


def forward(batch, model, task):
    if 'segmentation' in task:
        x = batch['image']                                          # (B, C, H, W)
        y_pr = model(x)                                             # (B, num_cls, H, W)
        y_pr = einops.rearrange(y_pr, 'B cls H W -> (B H W) cls')   # (B*H*W, num_cls)
        y_gt = batch['label']
        y_gt = einops.rearrange(y_gt, 'B H W -> (B H W)')           # (B*H*W)

    elif 'recognition' in task:
        x = batch['image']                                          # (B, N, C, H, W)
        x = einops.rearrange(x, 'B N C H W -> (B N) C H W')         # (B*N, C, H, W)
        y_pr = model(x)                                             # (B*N, num_cls)
        y_gt = batch['label']                                       # (B, N)
        y_gt = einops.rearrange(y_gt, 'B N -> (B N)')               # (B*N)

    else:
        raise ValueError("Invalid task")

    return y_pr, y_gt


@torch.no_grad()
def evaluate(dataloader, model, device, criterion, args):
    x_list = []
    y_pr_list = []
    y_gt_list = []
    loss_list = []
    results = {}

    model = model.to(device)
    for t, batch in enumerate(tqdm(dataloader)):
        model.eval()

        # Load batch to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Forward the input by the model
        x_list.append(batch['image'])
        y_pr, y_gt = forward(batch, model, args.task)

        # Loss
        loss = criterion(y_pr, y_gt)
        loss_list.append(loss)

        # Predict
        y_pr = torch.argmax(y_pr, dim=1)

        y_pr_list.append(y_pr)
        y_gt_list.append(y_gt)


    # Metric
    x = torch.cat(x_list, dim=0)        # (N_all, C, H, W)
    y_pr = torch.cat(y_pr_list, dim=0)  # (N_all)
    y_gt = torch.cat(y_gt_list, dim=0)  # (N_all)

    accuracy = accuracy_score(y_gt.cpu(), y_pr.cpu())
    loss = torch.tensor(loss_list).mean().item()

    # Reshape to Image
    if 'segmentation' in args.task:
        y_pr = einops.rearrange(y_pr, '(All C H W) -> All C H W', C=1, H=args.H, W=args.W)
        y_gt = einops.rearrange(y_gt, '(All C H W) -> All C H W', C=1, H=args.H, W=args.W)
    elif 'recognition' in args.task:
        y_pr = einops.rearrange(y_pr, '(All N) -> All N', N=args.max_char)
        y_gt = einops.rearrange(y_gt, '(All N) -> All N', N=args.max_char)

        y_pr_char = []
        y_gt_char = []

        with open(args.decoder_file, 'r') as decoder_file:
            decoder = json.load(decoder_file)
            decoder_file.close()

        for pr, gt in zip(y_pr, y_gt):
            # pr of shape (N)
            y_pr_char.append("".join(decoder[str(token.item())] for token in pr))
            y_gt_char.append("".join(decoder[str(token.item())] for token in gt))
            pass

        y_pr, y_gt = y_pr_char, y_gt_char

    results['metric'] = {"loss": loss,
                         "accuracy": accuracy}
    results['y_pr'] = y_pr
    results['y_gt'] = y_gt
    results['x'] = x

    return results
