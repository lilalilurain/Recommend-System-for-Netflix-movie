import numpy as np

def rmse(y_true, y_pred):
    """
    Tính Root Mean Squared Error giữa giá trị thật và dự đoán.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def precision_recall_at_k(recommended, relevant, k=10):
    """
    Precision@K và Recall@K
    recommended : list các item hệ thống đề xuất
    relevant    : list các item thật sự user thích (rating >= threshold)
    """
    rec_k = recommended[:k]
    rec_set = set(rec_k)
    rel_set = set(relevant)

    tp = len(rec_set & rel_set)
    prec = tp / max(1, len(rec_k))
    rec = tp / max(1, len(rel_set))

    return prec, rec

def ndcg_at_k(recommended, relevant, k=10):
    """
    Normalized Discounted Cumulative Gain (NDCG@K)
    """
    rel_set = set(relevant)
    dcg = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        gain = 1.0 if item in rel_set else 0.0
        dcg += gain / np.log2(i + 1)

    ideal_gains = [1.0] * min(k, len(rel_set))
    idcg = sum([g / np.log2(i + 1) for i, g in enumerate(ideal_gains, start=1)])

    return float(dcg / idcg) if idcg > 0 else 0.0
