import pandas as pd

# ====== CẤU HÌNH ======
files = [
    r"C:\Users\gaube\Downloads\Netflix data\data1.txt",
    r"C:\Users\gaube\Downloads\Netflix data\data2.txt",
    r"C:\Users\gaube\Downloads\Netflix data\data3.txt"
]
output_file = r"C:\Users\gaube\Downloads\Netflix data\merged_data.txt"

print("📌 Bắt đầu gộp dataset...")

dfs = []
for file in files:
    print(f"📂 Đang đọc file: {file}")
    # Thêm header=None để pandas không hiểu nhầm dòng đầu là header
    df = pd.read_csv(file, names=["userId", "movieId", "rating", "date"], header=None)
    print(f"   ✅ Đọc được {len(df):,} dòng")
    dfs.append(df)

# Gộp tất cả lại
merged_df = pd.concat(dfs, ignore_index=True)

# Xuất ra file
merged_df.to_csv(output_file, index=False, header=True)

print(f"🎉 Hoàn thành! File đã lưu tại: {output_file}")
print(f"👉 Tổng số dòng: {len(merged_df):,}")
print("📌 5 dòng đầu:")
print(merged_df.head())
