<<<<<<< HEAD
## 1) Tính năng chính
- Lọc cộng tác (Collaborative Filtering) bằng Matrix Factorization (Funk-SVD, SGD) + bias.
- Hệ gợi ý lai (Hybrid): kết hợp CF với nội dung (TF‑IDF từ genres/tags) bằng trọng số `alpha`.
- Mô-đun giả lập tấn công (shilling): Random / Average / Bandwagon.
- Phát hiện đánh giá giả (unsupervised/supervised): IsolationForest hoặc RandomForest trên đặc trưng hành vi người dùng.
- Giảm nhiễu: lọc hoặc down-weight người dùng bị nghi ngờ trước khi huấn luyện.
- Đánh giá: RMSE (xếp hạng), Precision@K, Recall@K, NDCG@K (top‑N) và so sánh có/không có bước phát hiện/làm sạch.

## 3) Cài đặt môi trường
```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
## 4) Chạy nhanh (pipeline đầy đủ)
```bash
# chạy toàn bộ: chuẩn bị -> (tuỳ chọn) giả lập tấn công -> phát hiện giả -> train CF/Hybrid -> đánh giá & so sánh
python main.py \
--data_dir data/ml-latest-small \
--mode full \
--simulate_attacks 1 --attack_type bandwagon --n_attack_users 200 --filler_ratio 0.05 --target_movie_id 1 \
--detector isolation_forest --suspicious_action filter \
--recommender hybrid --alpha 0.7 \
--topk 10
```
## 5) Cấu trúc dự án
```
.
├── main.py
├── requirements.txt
├── recsys/
│ ├── __init__.py
│ ├── data.py
│ ├── splits.py
│ ├── utils.py
│ ├── metrics.py
│ ├── mf.py
│ ├── content.py
│ ├── hybrid.py
│ ├── attacks.py
│ ├── features.py
│ ├── detect.py
│ └── evaluate.py
└── outputs/
```


## 6) Gợi ý thí nghiệm
- So sánh `attack_type` = random/average/bandwagon với các `n_attack_users` khác nhau.
- So sánh `suspicious_action` = filter (loại bỏ) vs downweight (giảm trọng số) với hệ số `downweight=0.2`.
- So sánh `recommender` = `cf` vs `hybrid` (thay `alpha`).
- Thử detector `isolation_forest` (unsupervised) vs `random_forest` (supervised — nếu có nhãn từ mô-đun giả lập).


---

