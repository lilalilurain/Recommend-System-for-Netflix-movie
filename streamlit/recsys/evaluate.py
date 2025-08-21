import numpy as np
import pandas as pd
from .metrics import rmse, precision_recall_at_k, ndcg_at_k

def evaluate_explicit_rmse(model, test_df: pd.DataFrame):
    preds = []
    for _, row in test_df.iterrows():
        preds.append(model.predict_single(int(row['userId']), int(row['movieId'])))
    return rmse(test_df['rating'].values, preds)


def evaluate_topk(model, train_df: pd.DataFrame, test_df: pd.DataFrame,
                  k=10, like_threshold=4.0):
    # Build train items per user để loại bỏ item đã xem
    train_items = train_df.groupby('userId')['movieId'].apply(set).to_dict()

    # Các item "relevant" trong test (rating >= like_threshold)
    test_likes = test_df[test_df['rating'] >= like_threshold]
    test_rel = test_likes.groupby('userId')['movieId'].apply(list).to_dict()

    # Toàn bộ item (train + test)
    all_items = pd.concat([train_df['movieId'], test_df['movieId']]).unique()

    precs, recs, ndcgs = [], [], []
    for uid, rel_items in test_rel.items():
        seen = train_items.get(uid, set())
        candidates = [m for m in all_items if m not in seen]

        # Nếu model là hybrid thì cần user_profile_item_ids
        if hasattr(model, 'cb'):
            scores = model.score_items(uid, candidates,
                                       user_profile_item_ids=list(seen))
        else:
            scores = model.score_items(uid, candidates)

        order = np.argsort(scores)[::-1]
        ranked = [candidates[i] for i in order[:k]]

        p, r = precision_recall_at_k(ranked, rel_items, k)
        nd = ndcg_at_k(ranked, rel_items, k)

        precs.append(p)
        recs.append(r)
        ndcgs.append(nd)

    return (float(np.mean(precs) if precs else 0.0),
            float(np.mean(recs) if recs else 0.0),
            float(np.mean(ndcgs) if ndcgs else 0.0))
