# recsys/mf.py
from __future__ import annotations
import numpy as np
from tqdm import tqdm


class MF:
    """
    Matrix Factorization (SGD mini-batch, vectorized)

    - Reindex userId/movieId -> [0..n-1] để P,Q gọn & truy cập nhanh
    - Mini-batch updates (np.add.at) – không cần vòng lặp Python theo từng phần tử
    - Ổn định: clip |err|, clip L2-norm từng hàng của P,Q
    - Early stopping theo validation RMSE

    API:
      fit(ratings[, sample_weights])
      predict(users, items) -> np.ndarray
      predict_single(user, item) -> float
      score_items(user, item_ids) -> np.ndarray
      recommend(user, n=10, exclude=None) -> (item_ids, scores)
    """

    def __init__(
        self,
        n_factors: int = 20,
        lr: float = 0.005,
        reg: float = 0.05,
        n_epochs: int = 10,
        min_rating: float = 1.0,
        max_rating: float = 5.0,
        random_state: int = 42,
        batch_size: int = 131072,   # ~128k interactions/batch
        max_err_abs: float = 5.0,   # clip |err|
        max_row_norm: float = 5.0,  # clip ||row||_2 of P,Q
        val_frac: float = 0.02,     # 2% for validation
        patience: int = 2,          # early-stop patience
    ):
        self.n_factors = int(n_factors)
        self.lr = float(lr)
        self.reg = float(reg)
        self.n_epochs = int(n_epochs)
        self.min_rating = float(min_rating)
        self.max_rating = float(max_rating)
        self.random_state = int(random_state)
        self.batch_size = int(batch_size)
        self.max_err_abs = float(max_err_abs)
        self.max_row_norm = float(max_row_norm)
        self.val_frac = float(val_frac)
        self.patience = int(patience)

        # set ở fit()
        self.P: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self._uid_map: dict[int, int] | None = None
        self._iid_map: dict[int, int] | None = None
        self._n_users: int = 0
        self._n_items: int = 0

    # ---------- utils ----------
    @staticmethod
    def _clip_row_norm(mat: np.ndarray, max_norm: float) -> None:
        """Clip L2-norm của từng hàng về tối đa max_norm (in-place)."""
        if max_norm <= 0:
            return
        eps = 1e-12
        norms = np.linalg.norm(mat, axis=1, keepdims=True)  # (n,1)
        factors = np.clip(max_norm / (norms + eps), a_min=None, a_max=1.0)
        mat *= factors

    def _encode_ids(self, users_raw: np.ndarray, items_raw: np.ndarray):
        """
        Reindex ID gốc -> code liên tục [0..n-1] và lưu mapping để predict.
        """
        u_uniques, u_codes = np.unique(users_raw, return_inverse=True)
        i_uniques, i_codes = np.unique(items_raw, return_inverse=True)
        self._uid_map = {int(uid): int(code) for code, uid in enumerate(u_uniques)}
        self._iid_map = {int(iid): int(code) for code, iid in enumerate(i_uniques)}
        return (
            u_codes.astype(np.int64),
            i_codes.astype(np.int64),
            int(len(u_uniques)),
            int(len(i_uniques)),
        )

    # ---------- training ----------
    def fit(self, ratings, sample_weights=None):
        """
        ratings: DataFrame có cột userId, movieId, rating
        sample_weights: array-like (cùng độ dài) hoặc None
        """
        rng = np.random.default_rng(self.random_state)

        users_raw = ratings["userId"].to_numpy(dtype=np.int64, copy=False)
        items_raw = ratings["movieId"].to_numpy(dtype=np.int64, copy=False)
        y_all = ratings["rating"].to_numpy(dtype=np.float32, copy=False)

        if sample_weights is None:
            w_all = np.ones_like(y_all, dtype=np.float32)
        else:
            w_all = np.asarray(sample_weights, dtype=np.float32)
            if w_all.shape[0] != y_all.shape[0]:
                raise ValueError("sample_weights phải cùng chiều với ratings.")

        u, i, n_users, n_items = self._encode_ids(users_raw, items_raw)
        self._n_users, self._n_items = n_users, n_items

        # khởi tạo nhỏ, float32
        self.P = (0.01 * rng.standard_normal((n_users, self.n_factors))).astype(np.float32)
        self.Q = (0.01 * rng.standard_normal((n_items, self.n_factors))).astype(np.float32)

        N = y_all.shape[0]
        idx = np.arange(N, dtype=np.int64)

        # validation split
        val_size = max(1, int(self.val_frac * N))
        rng.shuffle(idx)
        val_idx, tr_idx = idx[:val_size], idx[val_size:]

        best_val = np.inf
        bad_epochs = 0

        for epoch in range(self.n_epochs):
            rng.shuffle(tr_idx)

            # --- train mini-batch ---
            for start in range(0, tr_idx.size, self.batch_size):
                sl = tr_idx[start:start + self.batch_size]
                ub = u[sl]
                ib = i[sl]
                rb = y_all[sl]
                wb = w_all[sl]

                # predict (B,)
                preds = np.sum(self.P[ub] * self.Q[ib], axis=1, dtype=np.float32)
                preds = np.clip(preds, self.min_rating, self.max_rating)

                # error (B,1) + clip
                err = np.clip(wb * (rb - preds), -self.max_err_abs, self.max_err_abs).astype(np.float32)
                err = err[:, None]

                # gradients
                grad_P = err * self.Q[ib] - self.reg * self.P[ub]
                grad_Q = err * self.P[ub] - self.reg * self.Q[ib]

                # scatter-add
                np.add.at(self.P, ub, self.lr * grad_P)
                np.add.at(self.Q, ib, self.lr * grad_Q)

            # ổn định
            self._clip_row_norm(self.P, self.max_row_norm)
            self._clip_row_norm(self.Q, self.max_row_norm)

            # --- validation ---
            vub, vib = u[val_idx], i[val_idx]
            vpred = np.sum(self.P[vub] * self.Q[vib], axis=1, dtype=np.float32)
            vpred = np.clip(vpred, self.min_rating, self.max_rating)
            verr = y_all[val_idx] - vpred
            val_rmse = float(np.sqrt(np.mean(verr * verr)))
            tqdm.write(f"Epoch {epoch+1}/{self.n_epochs} | val RMSE: {val_rmse:.4f}")

            if val_rmse + 1e-5 < best_val:
                best_val = val_rmse
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    tqdm.write("Early stopping.")
                    break

        return self

    # ---------- inference ----------
    def _map_array(self, arr: np.ndarray, mapping: dict | None, size: int) -> np.ndarray:
        """Map ID gốc -> code; ID lạ map về 0 để tránh lỗi chỉ số."""
        if mapping is None:
            raise RuntimeError("Model chưa fit hoặc mapping chưa sẵn sàng.")
        out = np.fromiter((mapping.get(int(x), -1) for x in arr), count=arr.size, dtype=np.int64)
        out[out < 0] = 0
        return out

    def predict(self, users, items):
        """
        users, items: array-like ID gốc.
        Trả về: numpy array dự đoán (đã clip về [min_rating, max_rating]).
        """
        if self.P is None or self.Q is None:
            raise RuntimeError("Model chưa fit.")
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        ub = self._map_array(users, self._uid_map, self._n_users)
        ib = self._map_array(items, self._iid_map, self._n_items)
        preds = np.sum(self.P[ub] * self.Q[ib], axis=1, dtype=np.float32)
        return np.clip(preds, self.min_rating, self.max_rating)

    def predict_single(self, user: int, item: int) -> float:
        """Dự đoán cho một (user, item) dùng ID gốc."""
        return float(self.predict([user], [item])[0])

    def score_items(self, user_id: int, item_ids):
        """
        Trả về điểm dự đoán cho 1 user trên danh sách item_ids (ID gốc).
        Khớp với signature evaluate_topk(uid, candidates).
        """
        if self.P is None or self.Q is None:
            raise RuntimeError("Model chưa fit.")
        u_code = self._uid_map.get(int(user_id), 0)
        items = np.asarray(item_ids, dtype=np.int64)
        i_codes = np.fromiter((self._iid_map.get(int(x), 0) for x in items),
                              count=items.size, dtype=np.int64)
        user_vec = self.P[u_code]                          # (F,)
        scores = (user_vec[None, :] * self.Q[i_codes]).sum(axis=1, dtype=np.float32)
        return np.clip(scores, self.min_rating, self.max_rating)

    def recommend(self, user_id: int, n: int = 10, exclude=None):
        """
        Gợi ý top-n item cho user_id.
        exclude: iterable các item (ID gốc) cần loại bỏ (đã xem, v.v.)
        Trả về (item_ids, scores) đã sắp xếp giảm dần.
        """
        if self.P is None or self.Q is None:
            raise RuntimeError("Model chưa fit.")

        u_code = self._uid_map.get(int(user_id), 0)
        user_vec = self.P[u_code]                          # (F,)
        scores = (self.Q @ user_vec).astype(np.float32)    # (n_items,)
        scores = np.clip(scores, self.min_rating, self.max_rating)

        # map ngược code -> raw itemId (tạo nhanh – có thể cache nếu dùng nhiều)
        inv_i_map = np.empty(self._n_items, dtype=np.int64)
        for raw, code in self._iid_map.items():
            inv_i_map[code] = raw

        if exclude:
            exclude = set(int(x) for x in exclude)
            mask = np.ones(self._n_items, dtype=bool)
            for raw in exclude:
                code = self._iid_map.get(raw)
                if code is not None:
                    mask[code] = False
            scores = np.where(mask, scores, -1e9)

        n = max(1, min(n, scores.size))
        top_idx = np.argpartition(-scores, kth=n-1)[:n]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return inv_i_map[top_idx].tolist(), scores[top_idx].tolist()
