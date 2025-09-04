import pandas as pd

# Đọc dữ liệu
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")

#Chuyển các cột thời gian của df1 sang datetime
dt_cols = ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]
df1[dt_cols] = df1[dt_cols].apply(
    pd.to_datetime, format="%d/%m/%Y %H:%M", errors="coerce"
)

#Chuyển cột time của df2
df2["time"] = pd.to_datetime(df2["time"], format="%d/%m/%Y %H:%M", errors="coerce")

#Tạo thêm cột 'date' từ 'start_time'
df1["date"] = df1["start_time"].dt.date

# Tạo kiểu số cho các cột số
df1_numeric = ["bat_landing_to_food", "seconds_after_rat_arrival", "risk", "reward", "hours_after_sunset"]
df2_numeric = ["hours_after_sunset", "bat_landing_number", "food_availability", "rat_minutes", "rat_arrival_number"]

df1[df1_numeric] = df1[df1_numeric].apply(pd.to_numeric, errors="coerce")
df2[df2_numeric] = df2[df2_numeric].apply(pd.to_numeric, errors="coerce")

# Chuẩn hóa 'hours_after_sunset' về bậc 0.5 giờ (Dataset 1)
df1["date"] = pd.to_datetime(df1["start_time"].dt.date)
df1["hours_after_sunset"] = (df1["hours_after_sunset"] * 2).round() / 2

# Sửa lỗi chính tả / chuẩn hóa giá trị nhãn
df1["habit"] = df1["habit"].replace({
    "bat_figiht": "bat_fight",
    "rat attack": "rat_attack"
})
# Ràng buộc logic: risk, reward chỉ 0/1; season 0/1 
# risk / reward / season
df1 = df1[df1["risk"].isin([0, 1]) | df1["risk"].isna()]
df1 = df1[df1["reward"].isin([0, 1]) | df1["reward"].isna()]
df1 = df1[df1["season"].isin([0, 1]) | df1["season"].isna()]

# Loại "noise" trong cột habit và loại mẫu không có habit hợp lệ
noise_values = ["other", "others", "other_bats", "other bat", "other directions", "not_sure_rat", "all_pick"]
df1 = df1[~df1["habit"].isin(noise_values)]
df1["habit_numbers"] = df1["habit"].astype(str).str.extract(
    r"^(\d+(?:\.\d+)?(?:,\d+(?:\.\d+)?)*$)"
)

df1 = df1.dropna(subset=["habit"])

# Loại giá trị âm ở Dataset 1&2 cho các cột đếm / thời lượng
nonneg_cols_df1 = ["bat_landing_to_food", "seconds_after_rat_arrival", "risk", "reward"]
mask_nonneg_df1 = (df1[nonneg_cols_df1] >= 0) | df1[nonneg_cols_df1].isna()
df1 = df1[mask_nonneg_df1.all(axis=1)]
nonneg_cols_df2 = ["bat_landing_number", "food_availability", "rat_minutes", "rat_arrival_number"]
mask_nonneg_df2 = (df2[nonneg_cols_df2] >= 0) | df2[nonneg_cols_df2].isna()
df2 = df2[mask_nonneg_df2.all(axis=1)]

# Lọc giá trị hợp lý cho 'bat_landing_to_food' (0–60 giây) và theo IQR
Q1 = df1["bat_landing_to_food"].quantile(0.25)
Q3 = df1["bat_landing_to_food"].quantile(0.75)
IQR = Q3 - Q1
df1 = df1[
    (df1["bat_landing_to_food"] >= 0) &
    (df1["bat_landing_to_food"] <= 60) &
    (df1["bat_landing_to_food"] >= Q1 - 1.5 * IQR) &
    (df1["bat_landing_to_food"] <= Q3 + 1.5 * IQR)
]
# Xóa trùng lặp hoàn toàn
df1 = df1.drop_duplicates()
df2 = df2.drop_duplicates()

# Tạo kiểu category cho một số cột phân loại
df1[["habit", "month"]] = df1[["habit", "month"]].astype("category")

# Xuất dữ liệu sạch
df1.to_csv("dataset1_clean.csv", index=False)
df2.to_csv("dataset2_clean.csv", index=False)

