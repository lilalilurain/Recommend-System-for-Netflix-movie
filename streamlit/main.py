import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from recsys.data import load_netflix_prize
from recsys.splits import temporal_user_split
from recsys.attacks import inject_shilling_attacks
from recsys.features import build_user_features
from recsys.detect import train_unsupervised_iforest, train_supervised_rf
from recsys.mf import MF
from recsys.content import ContentModel   # ✅ đồng bộ
from recsys.hybrid import HybridModel
from recsys.evaluate import evaluate_explicit_rmse, evaluate_topk
from recsys.utils import set_seed, ensure_dir

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score
)


def run_pipeline(args, return_results=False):
    """
    Nếu return_results=True → trả về dict results + dict figs để Streamlit hiển thị.
    Nếu False → chỉ chạy như cũ, in ra console và lưu file.
    """
    set_seed(args.seed)
    ensure_dir(Path('outputs'))

    # 1. Load dữ liệu
    print("🔹 Bước 1: Load dữ liệu...")
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    ratings_cache = os.path.join(cache_dir, "ratings.pkl")
    movies_cache = os.path.join(cache_dir, "movies.pkl")

    if os.path.exists(ratings_cache):
        ratings = pd.read_pickle(ratings_cache)
        movies = pd.read_pickle(movies_cache) if os.path.exists(movies_cache) else None
        print(f"✅ Đã load dữ liệu cache: {len(ratings)} ratings")
    else:
        ratings, movies, _ = load_netflix_prize(args.data_dir, nrows=args.nrows)
        ratings.to_pickle(ratings_cache)
        if movies is not None:
            movies.to_pickle(movies_cache)
        print(f"✅ Đã load từ gốc: {len(ratings)} ratings")

    if ratings is None or len(ratings) == 0:
        raise ValueError("❌ Ratings data bị rỗng!")

    # 2. Chia train/test
    print("🔹 Bước 2: Chia train/test...")
    train_df, test_df = temporal_user_split(ratings, test_ratio=args.test_ratio)
    print(f"✅ Train: {len(train_df)}, Test: {len(test_df)}")

    # 3. Giả lập attack
    print("🔹 Bước 3: Giả lập attack...")
    y_attack = None
    if args.simulate_attacks:
        train_df, y_attack = inject_shilling_attacks(
            train_df, movies,
            attack_type=args.attack_type,
            n_attack_users=args.n_attack_users,
            filler_ratio=args.filler_ratio,
            target_movie_id=args.target_movie_id,
            rating_scale=(args.min_rating, args.max_rating)
        )
        print(f"✅ Đã thêm {args.n_attack_users} user tấn công")
    else:
        print("➡️ Bỏ qua bước attack")

    # 4. User features
    print("🔹 Bước 4: Trích xuất user features...")
    user_feats = build_user_features(train_df, movies)
    print(f"✅ Số user features: {user_feats.shape}")

    # 5. Phát hiện user đáng ngờ
    print("🔹 Bước 5: Phát hiện user đáng ngờ...")
    suspicious_users, y_true, y_pred = None, None, None
    if args.detector == 'isolation_forest':
        suspicious_users, y_pred = train_unsupervised_iforest(
            user_feats,
            contamination=args.contamination,
            return_labels=True
        )
        y_true = np.zeros_like(y_pred)
    elif args.detector == 'random_forest':
        if y_attack is None:
            suspicious_users, y_pred = train_unsupervised_iforest(
                user_feats,
                contamination=args.contamination,
                return_labels=True
            )
            y_true = np.zeros_like(y_pred)
        else:
            y_true = np.array([y_attack.get(uid, 0) for uid in user_feats.index])
            suspicious_users, y_true, y_pred = train_supervised_rf(
                user_feats, y_true, return_labels=True
            )
    else:
        raise ValueError("detector phải là isolation_forest hoặc random_forest")
    print(f"✅ Phát hiện {len(suspicious_users)} user đáng ngờ")

    # 6. Chuẩn bị dữ liệu train
    print("🔹 Bước 6: Chuẩn bị dữ liệu train...")
    baseline_train = train_df.copy()
    if args.suspicious_action == 'filter':
        cleaned_train = train_df[~train_df['userId'].isin(suspicious_users)].copy()
        sample_weights = None
        print(f"✅ Đã filter, còn {len(cleaned_train)} samples")
    elif args.suspicious_action == 'downweight':
        cleaned_train = train_df.copy()
        cleaned_train['weight'] = np.where(cleaned_train['userId'].isin(suspicious_users),
                                           args.downweight, 1.0)
        sample_weights = cleaned_train['weight'].values
        print("✅ Đã áp dụng downweight")
    else:
        raise ValueError("suspicious_action phải là filter | downweight")

    # 7. Train models
    print("🔹 Bước 7: Train models...")
    mf_base = MF(n_factors=args.factors, lr=args.lr, reg=args.reg,
                 n_epochs=args.epochs, min_rating=args.min_rating, max_rating=args.max_rating)
    mf_base.fit(baseline_train)
    print("✅ Đã train MF baseline")

    mf_clean = MF(n_factors=args.factors, lr=args.lr, reg=args.reg,
                  n_epochs=args.epochs, min_rating=args.min_rating, max_rating=args.max_rating)
    mf_clean.fit(cleaned_train, sample_weights=sample_weights)
    print("✅ Đã train MF cleaned")

    content = ContentModel()
    content.fit(movies, None)
    print("✅ Đã train Content model")

    if args.recommender == 'cf':
        model_base, model_clean = mf_base, mf_clean
        print("➡️ Sử dụng CF")
    elif args.recommender == 'hybrid':
        model_base = HybridModel(mf_base, content, alpha=args.alpha)
        model_clean = HybridModel(mf_clean, content, alpha=args.alpha)
        print("➡️ Sử dụng Hybrid")
    else:
        raise ValueError("recommender phải là cf | hybrid")

    # 8. Đánh giá
    print("🔹 Bước 8: Đánh giá...")
    rmse_base = evaluate_explicit_rmse(model_base, test_df)
    prec_b, rec_b, ndcg_b = evaluate_topk(model_base, baseline_train, test_df,
                                          k=args.topk, like_threshold=args.like_threshold)
    print(f"✅ Baseline RMSE={rmse_base:.4f}")

    rmse_clean = evaluate_explicit_rmse(model_clean, test_df)
    prec_c, rec_c, ndcg_c = evaluate_topk(model_clean, cleaned_train, test_df,
                                          k=args.topk, like_threshold=args.like_threshold)
    print(f"✅ Cleaned RMSE={rmse_clean:.4f}")

    # 9. Confusion matrix & metrics
    print("🔹 Bước 9: Confusion matrix & metrics...")
    figs = {}
    if y_true is not None and y_pred is not None:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plt.figure(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
        disp.plot(cmap="Blues", values_format="d", ax=plt.gca(), colorbar=False)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        cm_path = "outputs/confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)
        figs["confusion_matrix"] = cm_path
        plt.close()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        metrics_names = ["Accuracy", "Precision", "Recall"]
        values = [acc, prec, rec]
        plt.figure(figsize=(6, 5))
        plt.bar(metrics_names, values, color=["steelblue", "orange", "green"])
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("Classification Metrics")
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
        plt.tight_layout()
        clf_path = "outputs/classification_metrics.png"
        plt.savefig(clf_path, dpi=150)
        figs["classification_metrics"] = clf_path
        plt.close()
        print("✅ Đã vẽ Confusion matrix & Metrics")

    # 10. Kết quả
    print("🔹 Bước 10: Tổng hợp kết quả...")
    results = {
        "Baseline": {"RMSE": float(rmse_base),
                     f"Precision@{args.topk}": float(prec_b),
                     f"Recall@{args.topk}": float(rec_b),
                     f"NDCG@{args.topk}": float(ndcg_b)},
        "Cleaned": {"RMSE": float(rmse_clean),
                    f"Precision@{args.topk}": float(prec_c),
                    f"Recall@{args.topk}": float(rec_c),
                    f"NDCG@{args.topk}": float(ndcg_c)},
        "n_suspicious_users": int(len(suspicious_users)) if suspicious_users is not None else 0
    }

    if not return_results:
        with open("outputs/metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print("✅ Đã lưu kết quả tại outputs/metrics.json")
        return None
    else:
        return results, figs


# Entry CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--simulate_attacks", action="store_true", default=True)
    parser.add_argument("--attack_type", type=str, default="random")
    parser.add_argument("--n_attack_users", type=int, default=200)
    parser.add_argument("--filler_ratio", type=float, default=0.1)
    parser.add_argument("--target_movie_id", type=int, default=1)
    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)
    parser.add_argument("--detector", type=str, default="random_forest")
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--suspicious_action", type=str, default="filter")
    parser.add_argument("--downweight", type=float, default=0.5)
    parser.add_argument("--factors", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--recommender", type=str, default="cf")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--like_threshold", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_pipeline(args, return_results=False)
