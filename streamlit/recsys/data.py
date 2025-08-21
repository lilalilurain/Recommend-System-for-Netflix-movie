# recsys/data.py
from __future__ import annotations
import os
from typing import Optional, Tuple, List
import pandas as pd


CANDIDATE_FILENAMES: List[str] = ["merged_data.txt", "merged_data.csv"]


def _resolve_path(data_dir_or_file: str) -> str:
    """Trả về đường dẫn tuyệt đối tới file merged (txt/csv)."""
    p = os.path.abspath(data_dir_or_file)

    # Truyền trực tiếp file
    if os.path.isfile(p):
        return p

    # Truyền thư mục: thử ngay trong thư mục
    for name in CANDIDATE_FILENAMES:
        cand = os.path.join(p, name)
        if os.path.exists(cand):
            return os.path.abspath(cand)

    # Dò trong thư mục con
    for root, _, files in os.walk(p):
        for name in CANDIDATE_FILENAMES:
            if name in files:
                return os.path.abspath(os.path.join(root, name))

    raise FileNotFoundError(f"❌ Không tìm thấy merged_data (.txt/.csv) bên dưới: {p}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {
        "UserID": "userId", "userID": "userId", "USERID": "userId",
        "MovieID": "movieId", "movieID": "movieId", "MOVIEID": "movieId",
        "Rating": "rating", "RATING": "rating",
        "Date": "date", "DATE": "date", "Timestamp": "date", "timestamp": "date",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)
    return df


def load_netflix_prize(data_dir: str, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, None]:
    """
    Đọc dữ liệu Netflix Prize (file merged lớn) theo CHUNKS để tránh tràn RAM.
    - data_dir: thư mục chứa file hoặc đường dẫn file trực tiếp.
    - nrows: nếu đặt, chỉ đọc tối đa nrows dòng (hữu ích để test nhanh).
    Trả về: ratings, movies, None
    """
    file_path = _resolve_path(data_dir)
    print(f"📂 Đang đọc file: {file_path}")

    # Định nghĩa dtype nhẹ để giảm RAM ngay từ lúc parse
    # (nếu file có header khác sẽ được chuẩn hoá sau)
    dtype_hint = {
        "userId": "int32",
        "movieId": "int32",
        "rating": "float32",
    }

    # Đọc theo chunks
    chunksize = 1_000_000  # 1 triệu dòng/chunk (điều chỉnh nếu cần)
    reader = pd.read_csv(
        file_path,
        chunksize=chunksize,
        low_memory=True,
        engine="c",
        iterator=True,
    )

    chunks: List[pd.DataFrame] = []
    total = 0
    for chunk in reader:
        # Chuẩn hoá tên cột và map biến thể
        chunk = _normalize_columns(chunk)

        # Nếu thiếu cột bắt buộc -> báo lỗi sớm
        required = {"userId", "movieId", "rating"}
        if not required.issubset(chunk.columns):
            raise ValueError(f"❌ Thiếu cột {sorted(required)} trong file. Columns: {chunk.columns.tolist()}")

        # Ép kiểu nhẹ cho các cột có thể có
        for col, dt in dtype_hint.items():
            if col in chunk.columns:
                chunk[col] = chunk[col].astype(dt, copy=False)

        # Parse "date" nếu có; để tiết kiệm, chỉ parse khi cột tồn tại
        if "date" in chunk.columns and not pd.api.types.is_datetime64_any_dtype(chunk["date"]):
            # errors='coerce' để không vỡ nếu có vài giá trị lạ
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")

        chunks.append(chunk)
        total += len(chunk)
        if nrows is not None and total >= nrows:
            # cắt bớt phần dư của chunk cuối để đúng nrows
            over = total - nrows
            if over > 0:
                chunks[-1] = chunks[-1].iloc[:-over]
            break

    if not chunks:
        raise ValueError("❌ File rỗng hoặc không đọc được dữ liệu.")

    ratings = pd.concat(chunks, ignore_index=True)

    # Nếu toàn bộ cột 'date' đều NaT (hoặc không có), loại bỏ để downstream tự xử lý thứ tự
    if "date" in ratings.columns and ratings["date"].isna().all():
        ratings.drop(columns=["date"], inplace=True)

    # Movies duy nhất
    movies = pd.DataFrame({"movieId": pd.Series(ratings["movieId"].unique(), dtype="int32")})

    # Thống kê nhanh
    n = len(ratings)
    n_users = ratings["userId"].nunique()
    n_items = ratings["movieId"].nunique()
    print(f"✅ Đọc {n:,} dòng | users: {n_users:,} | movies: {n_items:,}")
    if "date" in ratings.columns:
        try:
            dmin, dmax = ratings["date"].min(), ratings["date"].max()
            if pd.notna(dmin) and pd.notna(dmax):
                print(f"   → Khoảng thời gian: {dmin.date()} → {dmax.date()}")
        except Exception:
            pass

    return ratings, movies, None
