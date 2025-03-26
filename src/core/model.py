import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from eicu_model import eICUBackbone
from src.metrics import get_metrics_binary, get_metrics_multiclass


class MMLBackbone(nn.Module):
    def __init__(self, args, tokenizer=None):
        super(MMLBackbone, self).__init__()
        self.args = args

        if args.dataset == "eicu":
            self.model = eICUBackbone(
                embedding_size=args.embedding_size,
                dropout=args.dropout,
                num_classes=1,
            )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = []
        for i, batch in enumerate(tqdm(data_loader)):
            loss = self.model(**batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss.append(loss.item())
        return {"loss": np.mean(total_loss)}

    def eval_epoch(self, data_loader, bootstrap):
        self.model.eval()
        ids, ys, y_scores = [], [], []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                ids.extend(batch["id"])
                y = batch["label"].to(self.args.device)
                y_score, _ = self.model.inference(**batch)
                ys.append(y.cpu())
                y_scores.append(y_score.cpu())
        ids = np.array(ids)
        ys = torch.cat(ys, dim=0).numpy()
        y_scores = torch.cat(y_scores, dim=0).numpy()
        if self.args.num_classes == 1:
            results = get_metrics_binary(ys, y_scores, bootstrap=bootstrap)
            predictions = np.stack([ids, ys, y_scores], axis=1)
        else:
            results = get_metrics_multiclass(ys, y_scores, bootstrap=bootstrap)
            predictions = np.concatenate([np.stack([ids, ys], axis=1), y_scores], axis=1)
        return results, predictions
