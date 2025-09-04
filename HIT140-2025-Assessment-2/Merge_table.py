import pandas as pd

# Read the cleaned datasets
df1 = pd.read_csv("dataset1_clean.csv")
df2 = pd.read_csv("dataset2_clean.csv")

# Merge on 'date' and 'hours_after_sunset'
merged = pd.merge(
    df1[["date", "hours_after_sunset", "risk", "reward", "bat_landing_to_food",
         "seconds_after_rat_arrival", "habit"]],
    df2[["date", "hours_after_sunset", "food_availability", "rat_arrival_number", "bat_landing_number"]],
    on=["date", "hours_after_sunset"],
    how="left"   # or "inner", depending on analysis goal
)

# Export the merged dataset (optional)
merged.to_csv("merged_dataset.csv", index=False)
