import pandas as pd

# Read data
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")

# Convert datetime columns of df1
dt_cols = ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]
df1[dt_cols] = df1[dt_cols].apply(
    pd.to_datetime, format="%d/%m/%Y %H:%M", errors="coerce"
)

# Convert 'time' column of df2 to datetime
df2["time"] = pd.to_datetime(df2["time"], format="%d/%m/%Y %H:%M", errors="coerce")

# Create new column 'date' from 'start_time'
df1["date"] = df1["start_time"].dt.date

# Create new column 'date' from 'time'
df2["date"] = pd.to_datetime(df2["time"]).dt.date

# Convert numeric columns
df1_numeric = ["bat_landing_to_food", "seconds_after_rat_arrival", "risk", "reward", "hours_after_sunset"]
df2_numeric = ["hours_after_sunset", "bat_landing_number", "food_availability", "rat_minutes", "rat_arrival_number"]

df1[df1_numeric] = df1[df1_numeric].apply(pd.to_numeric, errors="coerce")
df2[df2_numeric] = df2[df2_numeric].apply(pd.to_numeric, errors="coerce")

# Normalize 'hours_after_sunset' to 0.5-hour intervals
df1["hours_after_sunset"] = (df1["hours_after_sunset"] * 2).round() / 2
df2["hours_after_sunset"] = (df2["hours_after_sunset"] * 2).round() / 2

# Fix typos / normalize labels
df1["habit"] = df1["habit"].replace({
    "bat_figiht": "bat_fight",
    "rat attack": "rat_attack"
})

# Logical constraint: risk and reward must be 0/1 
df1 = df1[df1["risk"].isin([0, 1]) | df1["risk"].isna()]
df1 = df1[df1["reward"].isin([0, 1]) | df1["reward"].isna()]

# Remove noise values in 'habit' and invalid rows
noise_values = [
    "other", "others", "other_bats", "other bat", 
    "other directions", "not_sure_rat", "all_pick", 
    "bowl_out", "no_food"
]
df1 = df1[~df1["habit"].isin(noise_values)]

# Clean 'habit' column: remove numeric-only values
habit_clean = (
    df1["habit"].astype(str)
    .str.strip()
    .str.replace(r"\s*;\s*", ",", regex=True)   # replace ; with ,
    .str.replace(r"\s+", "", regex=True)        # remove whitespace
)

# Define function to check if a list is numeric-only
def check_numlist(xs):
    return (
        len(xs) > 0
        and pd.to_numeric(pd.Series(xs), errors="coerce").notna().all()
    )

is_numlist = habit_clean.str.split(",").apply(check_numlist)

# Remove rows where 'habit' is numeric-only
df1 = df1[~is_numlist]

# Remove rows with missing 'habit'
df1 = df1.dropna(subset=["habit"])

# Remove negative values in Dataset 1 & 2
nonneg_cols_df1 = ["bat_landing_to_food", "seconds_after_rat_arrival", "risk", "reward"]
df1 = df1[df1[nonneg_cols_df1].apply(lambda row: ((row >= 0) | row.isna()).all(), axis=1)]

nonneg_cols_df2 = ["bat_landing_number", "food_availability", "rat_minutes", "rat_arrival_number"]
df2 = df2[df2[nonneg_cols_df2].apply(lambda row: ((row >= 0) | row.isna()).all(), axis=1)]


# Keep only valid values for 'bat_landing_to_food' (0–60 seconds)
df1 = df1[
    (df1["bat_landing_to_food"] >= 0) &
    (df1["bat_landing_to_food"] <= 60)
]

# Remove duplicates
df1 = df1.drop_duplicates()
df2 = df2.drop_duplicates()

# Convert 'habit' to category type (normalized lowercase)
df1["habit"] = (
    df1["habit"].astype(str).str.strip().str.lower().astype("category")
)

# --- Add column: with_rat (1 if rat_period_start & rat_period_end exist, else 0)
df1["with_rat"] = df1.apply(
    lambda row: 1 if pd.notna(row["rat_period_start"]) and pd.notna(row["rat_period_end"]) else 0,
    axis=1
)

# Export cleaned data
df1.to_csv("dataset1_clean.csv", index=False)
df2.to_csv("dataset2_clean.csv", index=False)
