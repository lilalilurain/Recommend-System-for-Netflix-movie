import pandas as pd
from collections import Counter

# ====== CẤU HÌNH ======
data_file = r"C:\Users\gaube\Downloads\Netflix data\combined_data_3.txt"
filtered_file = r"C:\Users\gaube\Downloads\Netflix data\data3.txt"

print("📌 Bắt đầu quá trình lọc dữ liệu...")
print("➡️ File gốc:", data_file)

# ====== BƯỚC 1: Đếm số rating cho mỗi movie và user ======
print("🔄 Đang đếm số lượng rating cho mỗi movie và user...")

movie_counter = Counter()
user_counter = Counter()

with open(data_file, "r") as f:
    current_movie = None
    for line in f:
        line = line.strip()
        if line.endswith(":"):
            current_movie = int(line[:-1])
        else:
            try:
                user, rating, date = line.split(",")
                movie_counter[current_movie] += 1
                user_counter[int(user)] += 1
            except:
                continue

print("✔️ Hoàn thành bước 1: Đếm xong.")
print("   -> Số lượng movie:", len(movie_counter))
print("   -> Số lượng user :", len(user_counter))

# ====== BƯỚC 2: Xác định danh sách hợp lệ ======
print("🔎 Đang lọc ra các movie có >= 50 rating và user có >= 20 rating...")

valid_movies = {m for m, c in movie_counter.items() if c >= 50}
valid_users = {u for u, c in user_counter.items() if c >= 20}

print("✔️ Hoàn thành bước 2.")
print("   -> Movie hợp lệ:", len(valid_movies))
print("   -> User hợp lệ :", len(valid_users))

# ====== BƯỚC 3: Lọc và ghi dữ liệu ======
print("💾 Đang lọc dữ liệu và ghi ra file:", filtered_file)

with open(data_file, "r") as f_in, open(filtered_file, "w") as f_out:
    current_movie = None
    for line in f_in:
        line = line.strip()
        if line.endswith(":"):
            current_movie = int(line[:-1])
        else:
            try:
                user, rating, date = line.split(",")
                user = int(user)
                rating = float(rating)

                if current_movie in valid_movies and user in valid_users:
                    f_out.write(f"{user},{current_movie},{rating},{date}\n")
            except:
                continue

print("🎉 Hoàn thành! File đã lưu tại:", filtered_file)
