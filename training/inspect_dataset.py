
from datasets import load_dataset
import pandas as pd

# Step 1: Load the dataset
ds = load_dataset("ogozcelik/english-fake-news-detection")
df = ds["train"].to_pandas()

print("----- DATASET OVERVIEW -----")
print("Total tweets:", len(df))
print("Columns:", df.columns.tolist())

# Step 2: Check missing values
print("\n----- MISSING VALUES -----")
print(df.isnull().sum())

# Step 3: Check duplicate tweets
print("\n----- DUPLICATE TWEETS -----")
print("Duplicate tweets:", df["tweet"].duplicated().sum())

# Step 4: Check label distribution
print("\n----- LABEL DISTRIBUTION -----")
print(df["label"].value_counts())

# Step 5: Calculate tweet length
df["tweet_length"] = df["tweet"].astype(str).str.split().str.len()

print("\n----- TWEET LENGTH -----")
print("Average words:", round(df["tweet_length"].mean(), 2))
print("Shortest tweet (words):", df["tweet_length"].min())
print("Longest tweet (words):", df["tweet_length"].max())

# Step 6: Display one example per label
print("\n----- EXAMPLE TWEETS -----")

for label in ["True", "False", "Other"]:
    example = df[df["label"] == label].iloc[0]
    print(f"\nLabel: {label}")
    print("Tweet:", example["tweet"][:300])
