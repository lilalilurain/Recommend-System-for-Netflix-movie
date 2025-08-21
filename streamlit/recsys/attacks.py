import numpy as np
import pandas as pd


def inject_shilling_attacks(train_df, movies,
                            attack_type="random",
                            n_attack_users=50,
                            filler_ratio=0.1,
                            target_movie_id=1,
                            rating_scale=(1, 5)):
    """
    Giả lập shilling attacks bằng cách chèn thêm user giả vào train_df.
    
    Args:
        train_df (pd.DataFrame): Dữ liệu ratings gốc (userId, movieId, rating, date).
        movies (pd.DataFrame): Thông tin phim (không dùng nhiều ở đây).
        attack_type (str): Kiểu attack ("random", "average").
        n_attack_users (int): Số lượng attacker cần chèn.
        filler_ratio (float): Tỉ lệ số phim filler mỗi attacker sẽ rating.
        target_movie_id (int): Phim mục tiêu bị tấn công (sẽ được rate cao).
        rating_scale (tuple): Thang điểm (min_rating, max_rating).

    Returns:
        train_new (pd.DataFrame): DataFrame train có thêm attacker.
        y_attack (dict): Dict {userId: 0 nếu normal, 1 nếu attacker}.
    """
    train_new = train_df.copy()
    all_users = train_df['userId'].unique().tolist()

    # Tạo userId mới cho attacker
    start_uid = max(all_users) + 1
    attacker_ids = list(range(start_uid, start_uid + n_attack_users))

    ratings = []
    min_rating, max_rating = rating_scale

    all_movies = train_df['movieId'].unique()
    n_filler = int(filler_ratio * len(all_movies))

    for uid in attacker_ids:
        filler_items = np.random.choice(all_movies, size=n_filler, replace=False)

        for mid in filler_items:
            if attack_type == "random":
                r = np.random.uniform(min_rating, max_rating)
            elif attack_type == "average":
                # trung bình rating của phim đó
                r = train_df[train_df['movieId'] == mid]['rating'].mean()
                if np.isnan(r):
                    r = np.random.uniform(min_rating, max_rating)
            else:
                raise ValueError("attack_type phải là random | average")

            ratings.append((uid, mid, r, "2005-01-01"))

        # Target movie luôn rate = max
        ratings.append((uid, target_movie_id, max_rating, "2005-01-01"))

    # Gộp attacker vào train gốc
    df_attack = pd.DataFrame(ratings, columns=["userId", "movieId", "rating", "date"])
    train_new = pd.concat([train_new, df_attack], ignore_index=True)

    # Build nhãn: attacker =1, normal =0
    y_attack = {uid: 0 for uid in train_new['userId'].unique()}
    for uid in attacker_ids:
        y_attack[uid] = 1

    # In thống kê gọn
    n_attack = sum(y_attack.values())
    n_normal = len(y_attack) - n_attack
    print(f"🔹 Số lượng user Attack được chèn: {len(attacker_ids)}")
    print(f"   → Tổng số user Attack trong nhãn: {n_attack}")
    print(f"   → Tổng số user Normal trong nhãn: {n_normal}")

    return train_new, y_attack
