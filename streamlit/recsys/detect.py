# recsys/detect.py
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ---------- helpers ----------

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lọc cột numeric, xử lý NaN/inf, scale bằng StandardScaler và
    TRẢ VỀ DataFrame MỚI dạng float32 (tránh gán chéo dtype gây warning).

    Raises
    ------
    ValueError: nếu df rỗng hoặc không có cột numeric.
    """
    if df is None or len(df) == 0:
        raise ValueError("User features bị rỗng.")

    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] == 0:
        raise ValueError("User features không có cột số.")

    # Làm sạch và ép về float64 trước khi scale (ổn định số học hơn)
    num = num.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled = scaler.fit_transform(num.to_numpy(dtype=np.float64))

    # Tạo DataFrame mới thay vì gán .loc để tránh FutureWarning "incompatible dtype"
    num_scaled = pd.DataFrame(scaled, index=num.index, columns=num.columns)
    return num_scaled.astype(np.float32)


# ---------- unsupervised: Isolation Forest ----------

def train_unsupervised_iforest(
    user_features: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 300,
    max_samples: Union[int, float, str] = "auto",
    random_state: int = 42,
    n_jobs: int = -1,
    return_labels: bool = False,
) -> Union[List, Tuple[List, np.ndarray]]:
    """
    Phát hiện user đáng ngờ bằng IsolationForest (unsupervised).

    Parameters
    ----------
    user_features : DataFrame
        Index là userId, cột là đặc trưng số của user.
    contamination : float
        Tỉ lệ outlier mong đợi.
    return_labels : bool
        Nếu True, trả về thêm y_pred (0/1).

    Returns
    -------
    suspicious_users : list
        Danh sách userId được dự đoán là tấn công (y_pred=1).
    y_pred : np.ndarray, optional
        Nhãn 0/1 cho toàn bộ user (1 = anomaly/attack) nếu return_labels=True.
    """
    X = _prepare_features(user_features)

    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    clf.fit(X)

    pred = clf.predict(X)          # {1: normal, -1: anomaly}
    y_pred = (pred == -1).astype(int)

    suspicious_users = user_features.index[y_pred == 1].tolist()
    if return_labels:
        return suspicious_users, y_pred
    return suspicious_users


# ---------- supervised: Random Forest (OOF) ----------

def train_supervised_rf(
    user_features: pd.DataFrame,
    y_true: Union[np.ndarray, List[int]],
    n_estimators: int = 400,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    n_splits: int = 5,
    threshold: float = 0.5,
    random_state: int = 42,
    n_jobs: int = -1,
    return_labels: bool = False,
) -> Union[List, Tuple[List, np.ndarray, np.ndarray]]:
    """
    Huấn luyện giám sát RandomForest với dự đoán OOF (out-of-fold) cho toàn bộ user.

    - Nếu tất cả nhãn đều 0 (không có attack) → trả về toàn 0.
    - Dùng class_weight='balanced' để xử lý mất cân bằng lớp.

    Returns
    -------
    suspicious_users : list
        userId có y_pred = 1.
    (y_true_aligned, y_pred) : tuple, optional
        Trả về khi return_labels=True.
    """
    if user_features is None or len(user_features) == 0:
        raise ValueError("User features bị rỗng.")

    y_true = np.asarray(y_true, dtype=int)
    if y_true.shape[0] != user_features.shape[0]:
        raise ValueError(
            "Độ dài y_true không khớp số hàng của user_features "
            f"({y_true.shape[0]} != {user_features.shape[0]})."
        )

    X = _prepare_features(user_features)
    n_samples = X.shape[0]

    # Không có positive → dự đoán toàn 0
    if int(np.sum(y_true == 1)) == 0:
        y_pred = np.zeros_like(y_true, dtype=int)
        suspicious_users = []
        if return_labels:
            return suspicious_users, y_true, y_pred
        return suspicious_users

    # Bảo đảm n_splits không vượt kích thước lớp nhỏ nhất
    min_class = int(np.min(np.bincount(y_true)))
    n_splits = int(max(2, min(n_splits, min_class)))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_proba = np.zeros(n_samples, dtype=np.float32)

    for tr_idx, val_idx in skf.split(X, y_true):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr = y_true[tr_idx]

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        clf.fit(X_tr, y_tr)

        proba_val = clf.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = proba_val.astype(np.float32)

    y_pred = (oof_proba >= float(threshold)).astype(int)
    suspicious_users = user_features.index[y_pred == 1].tolist()

    if return_labels:
        return suspicious_users, y_true, y_pred
    return suspicious_users
