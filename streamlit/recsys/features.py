import pandas as pd


def build_user_features(ratings: pd.DataFrame, movies: pd.DataFrame = None):
    """
    Xây dựng đặc trưng user từ ratings + movies (nếu có).
    Trả về DataFrame: userId -> các đặc trưng.
    """
    if movies is None:
        movies = pd.DataFrame({"movieId": ratings["movieId"].unique()})

    # Gộp dữ liệu ratings với movies (nếu có)
    df = ratings.merge(movies[["movieId"]], on="movieId", how="left")

    # Tính toán đặc trưng cơ bản theo user
    feats = df.groupby("userId").agg(
        n_ratings=("rating", "count"),
        mean_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        min_rating=("rating", "min"),
        max_rating=("rating", "max"),
    )

    # Điền NaN = 0 và đặt index = userId
    feats = feats.fillna(0)
    feats.index.name = "userId"

    return feats
