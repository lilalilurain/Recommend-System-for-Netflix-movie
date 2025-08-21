# recsys/splits.py
from __future__ import annotations
import numpy as np
import pandas as pd

def temporal_user_split(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    user_col: str = "userId",
    time_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chia train/test theo *thời gian trong từng user* mà không cần concat nhiều mảnh.
    Bước làm:
      1) Xác định cột thời gian (timestamp -> date -> tạo chỉ mục giả).
      2) Sort theo (user, time) bằng sort ổn định.
      3) Với mỗi user, dùng cumcount để lấy vị trí rồi cắt đuôi theo test_ratio.

    Trả về: train_df, test_df
    """
    if ratings is None or len(ratings) == 0:
        raise ValueError("ratings rỗng.")

    df = ratings.copy()

    # Chọn cột thời gian
    if time_col is None:
        if "timestamp" in df.columns:
            time_col = "timestamp"
        elif "date" in df.columns:
            time_col = "date"
        else:
            time_col = "_order_idx"
            df[time_col] = np.arange(len(df), dtype=np.int64)

    # Nếu là datetime, đổi sang int64 (epoch ns) để thao tác nhanh & ít RAM
    if np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = df[time_col].astype("int64")  # thay cho .view("int64") để tránh FutureWarning

    # Sắp xếp ổn định theo user rồi thời gian
    df = df.sort_values([user_col, time_col], kind="mergesort", ignore_index=True)

    # Chỉ số vị trí trong từng user
    pos = df.groupby(user_col).cumcount()

    # Tổng số bản ghi mỗi user
    cnt = df.groupby(user_col)[time_col].transform("size")

    # Số phần tử đầu mỗi user đi vào train
    split_idx = np.ceil(cnt * (1.0 - float(test_ratio))).astype(np.int64)

    is_train = pos < split_idx
    train_df = df[is_train].copy()
    test_df = df[~is_train].copy()

    return train_df, test_df
