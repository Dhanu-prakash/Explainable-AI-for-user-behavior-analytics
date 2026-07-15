import pandas as pd

df = pd.read_csv("dataset.csv")

# Drop unnecessary columns
df = df.drop(columns=["Customer ID", "City"])

# Convert categorical → numeric
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

df["Membership Type"] = df["Membership Type"].map({
    "Bronze": 0,
    "Silver": 1,
    "Gold": 2
})

df["Discount Applied"] = df["Discount Applied"].map({
    True: 1,
    False: 0
})

df["Satisfaction Level"] = df["Satisfaction Level"].map({
    "Unsatisfied": 0,
    "Neutral": 1,
    "Satisfied": 2
})

print(df.head())

df.to_csv("processed_data.csv", index=False)