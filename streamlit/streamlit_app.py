import streamlit as st
import argparse
from main import run_pipeline

st.set_page_config(page_title="Netflix Recommender", layout="wide")

st.title("🎬 Netflix Recommendation System")
st.write("Hệ gợi ý phim Netflix với phát hiện & giảm thiểu đánh giá giả (shilling attacks).")

# ======================
# Sidebar config
# ======================
st.sidebar.header("⚙️ Cấu hình")

data_dir = st.sidebar.text_input("Thư mục dữ liệu", "data/ml-latest-small")

# Shilling attacks
st.sidebar.subheader("Shilling Attack")
attack_type = st.sidebar.selectbox("Kiểu tấn công", ["random", "average", "bandwagon"])
n_attack_users = st.sidebar.slider("Số user tấn công", 50, 500, 200, 50)
filler_ratio = st.sidebar.slider("Filler ratio", 0.01, 0.5, 0.1, 0.01)
target_movie_id = st.sidebar.number_input("Target movie id", min_value=1, value=1)

# Detector
st.sidebar.subheader("Phát hiện giả")
detector = st.sidebar.selectbox("Detector", ["isolation_forest", "random_forest"])
contamination = st.sidebar.slider("Contamination", 0.01, 0.5, 0.05, 0.01)

# Suspicious handling
suspicious_action = st.sidebar.selectbox("Xử lý user nghi ngờ", ["filter", "downweight"])
downweight = st.sidebar.slider("Trọng số khi downweight", 0.1, 1.0, 0.5, 0.1)

# Recommender
st.sidebar.subheader("Mô hình gợi ý")
recommender = st.sidebar.selectbox("Recommender", ["lọc cộng tác", "phép lai hybrid"])
alpha = st.sidebar.slider("Alpha (chỉ dùng cho hybrid)", 0.0, 1.0, 0.5, 0.1)

# Model params
st.sidebar.subheader("Tham số huấn luyện")
factors = st.sidebar.slider("Latent factors", 10, 200, 20, 10)
lr = st.sidebar.number_input("Learning rate", 0.001, 0.1, 0.01, 0.001)
reg = st.sidebar.number_input("Regularization", 0.0, 0.1, 0.02, 0.01)
epochs = st.sidebar.slider("Epochs", 5, 50, 10, 5)

# Eval params
st.sidebar.subheader("Đánh giá")
topk = st.sidebar.slider("Top-K", 5, 20, 10, 1)
like_threshold = st.sidebar.slider("Ngưỡng like", 1.0, 5.0, 4.0, 0.5)

seed = st.sidebar.number_input("Random seed", value=42)

# ======================
# Run pipeline
# ======================
if st.sidebar.button("🚀 Run experiment"):
    args = argparse.Namespace(
        data_dir=data_dir,
        nrows=None,
        test_ratio=0.2,
        simulate_attacks=True,
        attack_type=attack_type,
        n_attack_users=n_attack_users,
        filler_ratio=filler_ratio,
        target_movie_id=target_movie_id,
        min_rating=1.0,
        max_rating=5.0,
        detector=detector,
        contamination=contamination,
        suspicious_action=suspicious_action,
        downweight=downweight,
        factors=factors,
        lr=lr,
        reg=reg,
        epochs=epochs,
        alpha=alpha,
        recommender=recommender,
        topk=topk,
        like_threshold=like_threshold,
        seed=seed
    )

    with st.spinner("⏳ Đang chạy pipeline..."):
        results, figs = run_pipeline(args, return_results=True)

    st.success("✅ Hoàn thành!")

    # Show metrics
    st.subheader("📊 Kết quả so sánh Baseline vs Cleaned")
    st.json(results)

    # Show figures
    if figs.get("confusion_matrix"):
        st.subheader("Confusion Matrix")
        st.image(figs["confusion_matrix"])

    if figs.get("classification_metrics"):
        st.subheader("Classification Metrics")
        st.image(figs["classification_metrics"])

else:
    st.info("👉 Chọn tham số ở sidebar rồi bấm **Run experiment** để chạy.")
