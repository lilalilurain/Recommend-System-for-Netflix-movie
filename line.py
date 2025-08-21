import pandas as pd

# Thay 'your_dataset.csv' bằng đường dẫn đến file của bạn
file_path = r'C:\Users\gaube\Downloads\Netflix data\combined_data_4.txt' 

try:
    # Đọc file CSV vào một DataFrame của pandas
    df = pd.read_csv(file_path)

    # Lấy số dòng của DataFrame
    num_rows = len(df)

    print(f"Số dòng của dataset là: {num_rows}")

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn '{file_path}'")
except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")