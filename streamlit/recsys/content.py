# content.py
import pandas as pd
import numpy as np

class ContentModel:
    def __init__(self):
        self.movie_features = None

    def fit(self, movies: pd.DataFrame = None, tags: pd.DataFrame = None):
        """
        Huấn luyện mô hình content-based.
        movies: DataFrame có ít nhất 'movieId', tùy chọn thêm ['title','year'].
        tags: DataFrame tùy chọn.
        """
        self.movie_features = self._prep_text(movies, tags)

    def _prep_text(self, movies, tags):
        if movies is None:
            # Nếu không có movies thì chỉ tạo movieId rỗng
            return pd.DataFrame(columns=["movieId"])

        base = movies.copy()

        # Nếu có cột title/year thì giữ lại, không thì bỏ qua
        keep_cols = ["movieId"]
        if "title" in base.columns:
            keep_cols.append("title")
        if "year" in base.columns:
            keep_cols.append("year")

        return base[keep_cols]

    def recommend(self, user_history, topk=10):
        """
        Trả về list movieId gợi ý (dummy content-based).
        """
        if self.movie_features is None or len(self.movie_features) == 0:
            return []

        # Giả sử recommend random từ tập movieId
        return self.movie_features["movieId"].sample(n=min(topk, len(self.movie_features))).tolist()

    def score_items(self, user_profile_item_ids, item_ids):
        """
        Cho điểm các item dựa trên content similarity (dummy).
        user_profile_item_ids: list các movie user đã xem.
        item_ids: list các movie cần tính điểm.
        """
        scores = []
        for item in item_ids:
            # Nếu item có trong profile thì score cao, ngược lại random
            if item in user_profile_item_ids:
                scores.append(1.0)   # giống
            else:
                scores.append(0.5)   # tạm random trung bình
        return np.array(scores, dtype=float)
