# app.py
import argparse
import os
from io import BytesIO, StringIO
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from main import run_pipeline

# =========================
# Page & constants
# =========================
st.set_page_config(page_title="Hệ thống gợi ý phim cho Netflix", page_icon="🎬", layout="wide")
st.title("🎬 Hệ thống đề xuất phim Netflix")
st.caption("Hệ thống gợi ý phim Netflix có khả năng phát hiện và giảm thiểu đánh giá giả (shilling attacks).")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Helpers
# =========================
def save_uploaded_file(uploaded_file, dest_path: Path) -> Path:
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest_path


@st.cache_data(show_spinner=False)
def preview_txt_from_bytes(file_bytes: bytes, n_preview: int = 50) -> pd.DataFrame | None:
    if not file_bytes:
        return None

    preview_data = file_bytes[:1 * 1024 * 1024]  # 1MB
    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']

    for encoding in encodings_to_try:
        try:
            txt_io = StringIO(preview_data.decode(encoding, errors="ignore"))
            separators = [",", "\t", ";", "::", "|", r'\s+']
            for sep in separators:
                try:
                    df = pd.read_csv(txt_io, sep=sep, nrows=n_preview, engine='python',
                                     header=None, names=["userID", "movieID", "rating", "date"])
                    if df.shape[1] == 4:
                        return df
                except Exception:
                    txt_io.seek(0)
                    continue
        except UnicodeDecodeError:
            continue
    return None


@st.cache_data(show_spinner=False)
def load_titles_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame | None:
    if not file_bytes:
        return None

    bio = BytesIO(file_bytes)
    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']

    for encoding in encodings_to_try:
        try:
            bio.seek(0)
            df = pd.read_csv(
                bio,
                header=None,
                names=["movieId", "year", "title"],
                usecols=[0, 1, 2],
                encoding=encoding
            )

            if df.shape[1] == 3:
                df["movieId"] = pd.to_numeric(df["movieId"], errors="coerce")
                df = df.dropna(subset=["movieId"]).astype({"movieId": "int64"})

                if "year" in df.columns:
                    df["year"] = pd.to_numeric(df["year"], errors="coerce")
                if "title" in df.columns:
                    df["title"] = df["title"].astype(str).str.strip()

                return df
        except Exception:
            continue

    st.warning("Không thể đọc file CSV. Vui lòng kiểm tra lại mã hóa và cấu trúc.")
    return None

# =========================
# Sidebar – upload dữ liệu
# =========================
st.sidebar.header("⚙️ Cấu hình dữ liệu")

dataset_file = st.sidebar.file_uploader(
    "Tải lên **merged.txt** (userID,movieID,rating,date) — đã nâng giới hạn upload trong config",
    type=["txt", "csv"],
)

titles_file = st.sidebar.file_uploader(
    "Tải lên **movie_tittles.csv** (movieId,year,title – KHÔNG header)",
    type=["csv"],
)

titles_df = load_titles_csv_from_bytes(titles_file.getvalue()) if titles_file else None
if titles_df is not None:
    st.sidebar.success(f"Đã đọc {len(titles_df):,} tiêu đề")
else:
    st.sidebar.info("Chưa tải titles.")

# =========================
# Main preview & tra cứu
# =========================
st.info("Upload **merged.txt** và **movie_tittles.csv** ở thanh bên, sau đó nhấn **Chạy thử nghiệm** để chạy.")

colA, colB = st.columns(2)
with colA:
    st.subheader("👀 Xem trước merged.txt")
    if dataset_file is not None:
        prev = preview_txt_from_bytes(dataset_file.getvalue())
        if prev is not None:
            st.dataframe(prev.head(20), use_container_width=True)
        else:
            st.caption("Không đọc được preview (file quá lớn hoặc định dạng không hợp lệ?).")
    else:
        st.caption("Chưa upload merged.txt.")

with colB:
    st.subheader("📋 Xem trước tiêu đề (movie_tittles.csv)")
    if titles_df is not None and not titles_df.empty:
        st.dataframe(titles_df.head(20), use_container_width=True)
    else:
        st.caption("Chưa upload hoặc file rỗng.")

# =========================
# Tra cứu phim
# =========================
st.subheader("🔍 Tra cứu phim")
if titles_df is not None and not titles_df.empty:
    tab1, tab2 = st.tabs(["Theo movieId", "Theo tên phim"])

    with tab1:
        q_id = st.number_input("Nhập movieId", min_value=1, value=1)
        found = titles_df.loc[titles_df["movieId"] == int(q_id)]
        if not found.empty:
            row = found.iloc[0]
            yr = "" if pd.isna(row.get("year")) else int(row.get("year"))
            st.success(f"🎥 {row['title']}{f' ({yr})' if yr else ''}")
        else:
            st.info("Không tìm thấy movieId này trong danh sách.")

    with tab2:
        q_title = st.text_input("Nhập tên phim", value="")
        if q_title.strip():
            matches = titles_df[titles_df["title"].str.lower().str.contains(q_title.strip().lower())]
            if not matches.empty:
                st.write(f"🔎 Tìm thấy {len(matches)} kết quả:")
                st.dataframe(matches[["movieId", "title", "year"]], use_container_width=True)
            else:
                st.warning("Không tìm thấy phim nào khớp với tên đã nhập.")
else:
    st.caption("Cần upload movie_tittles.csv để tra cứu.")

st.markdown("---")
with st.form("params_form"):
    st.subheader("⚙️ Tham số chạy thử nghiệm")

    col1, col2 = st.columns(2)
    with col1:
        recommender = st.selectbox("Thuật toán gợi ý", ["cf", "hybrid"])
        detector = st.selectbox("Bộ phát hiện", ["random_forest", "isolation_forest"])
        suspicious_action = st.selectbox("Xử lý user đáng ngờ", ["filter", "downweight"])
        simulate_attacks = st.checkbox("Giả lập tấn công", value=True)

    with col2:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
        factors = st.number_input("Số latent factors", min_value=1, max_value=200, value=20)
        downweight = st.slider("Trọng số downweight", 0.1, 1.0, 0.5)
        alpha = st.slider("Alpha (trong Hybrid)", 0.0, 1.0, 0.5)

    submitted = st.form_submit_button("🚀 Chạy thử nghiệm")

if submitted and dataset_file is not None:
    with st.spinner("Đang chạy pipeline..."):
        data_path = save_uploaded_file(dataset_file, UPLOAD_DIR / "merged.txt")

        args = argparse.Namespace(
            data_dir=UPLOAD_DIR,
            nrows=None,
            test_ratio=0.2,
            simulate_attacks=simulate_attacks,
            attack_type="random",
            n_attack_users=200,
            filler_ratio=0.1,
            target_movie_id=1,
            min_rating=1.0,
            max_rating=5.0,
            detector=detector,
            contamination=0.05,
            suspicious_action=suspicious_action,
            downweight=downweight,
            factors=factors,
            lr=0.01,
            reg=0.02,
            epochs=epochs,
            alpha=alpha,
            recommender=recommender,
            topk=10,
            like_threshold=4.0,
            seed=42,
        )

        results, figs = run_pipeline(args, return_results=True)
        st.success("✅ Đã chạy xong!")

        # Hiển thị kết quả
        st.json(results)

        if figs:
            for name, path in figs.items():
                st.image(path, caption=name)


# =========================
# Run pipeline
# =========================
st.warning("Cần bổ sung form nhập tham số cho pipeline trước khi chạy.")
