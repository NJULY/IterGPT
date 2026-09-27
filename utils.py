import numpy as np


def compute_metrics(ranks):
    ranks = np.array(ranks)
    metrics = {
        "hits@1": np.round(np.mean(ranks<=1), 3),
        "hits@3": np.round(np.mean(ranks<=3), 3),
        "hits@10": np.round(np.mean(ranks<=10), 3),
        "mrr": np.round(np.mean(1./ranks), 3),
    }
    return metrics

