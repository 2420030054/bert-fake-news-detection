
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
ds = load_dataset("ogozcelik/english-fake-news-detection")
df = ds["train"].to_pandas()

# Step 2: Keep only the columns we need
df = df[["tweet", "label"]].copy()
# Rename the dataset labels for our project
label_mapping = {
    "True": "true-positive",
    "False": "false-negative",
    "Other": "other-neutral"
}

df["label"] = df["label"].map(label_mapping)


# Step 3: Split into training and testing data
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Step 4: Display the results
print("Total records:", len(df))
print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print("\nTraining label distribution:")
print(train_df["label"].value_counts())

print("\nTesting label distribution:")
print(test_df["label"].value_counts())
 
# Step 5: Save the datasets
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("\nDatasets saved successfully!")
print("Training file: data/processed/train.csv")
print("Testing file: data/processed/test.csv")
