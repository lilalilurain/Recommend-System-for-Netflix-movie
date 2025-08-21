import numpy as np

class HybridModel:
    """
    Mô hình lai: kết hợp Collaborative Filtering (CF) và Content-based (CB).
    alpha = trọng số cho CF, (1 - alpha) cho CB.
    """

    def __init__(self, cf_model, content_model, alpha=0.7):
        self.cf = cf_model
        self.cb = content_model
        self.alpha = alpha

    def predict_single(self, userId, movieId):
        # với dự đoán rating ta ưu tiên CF
        return self.cf.predict_single(userId, movieId)

    def score_items(self, userId, item_ids, user_profile_item_ids=None):
        cf_scores = self.cf.score_items(userId, item_ids)

        if user_profile_item_ids is None:
            user_profile_item_ids = []

        cb_scores = self.cb.score_items(user_profile_item_ids, item_ids)

        # Chuẩn hóa điểm để kết hợp (z-score)
        def z(x):
            m = x.mean() if len(x) else 0
            s = x.std() if len(x) else 1
            return (x - m) / (s + 1e-6)

        return self.alpha * z(cf_scores) + (1 - self.alpha) * z(cb_scores)
