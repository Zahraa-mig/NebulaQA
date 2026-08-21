import pandas as pd
from datasets import load_dataset


print("Loading dataset from Hugging Face...")
ds = load_dataset("juliensimon/stackexchange-space-qa", split="train")
df = ds.to_pandas()

print(f"Number of rows before cleaning: {len(df):,}")
print(f"Number of columns: {df.shape[1]}")
print("\n Column Names:")
print(df.columns.tolist())


print("\n--- General Column Information ---")
print(df.info())

print("\n--- missing values count per column ---")
print(df.isnull().sum())

print("\n--- question distribution by source site ---")
print(df["site"].value_counts())


before = len(df)
df = df.dropna(subset=["answer_body", "answer_score", "answer_accepted"]).copy()
after = len(df)
print(f"\nDropped {before - after:,} missing or unanswered records.")
print(f"Number of rows after dropping missing values: {after:,}")



before = len(df)
df = df.drop_duplicates(subset=["qid", "site"]).copy()
print(f"Dropped {before - len(df):,} duplicate rows based on (qid, site).")
print(f"Remaining dataset size: {len(df):,}")



def clean_text(text: str) -> str:
    """Basic text normalization: removes leading/trailing spaces and condenses internal whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = " ".join(text.split())  
    return text


for col in ["question_title", "question_body", "answer_body"]:
    df[col] = df[col].apply(clean_text)


before = len(df)
df = df[(df["question_body"].str.len() > 0) & (df["answer_body"].str.len() > 0)]
print(f"Dropped {before - len(df):,} rows with empty text after normalization.")

before = len(df)
df = df[(df["question_body"].str.len() >= 15) & (df["answer_body"].str.len() >= 15)]
print(f"dropped {before - len(df):,} rows with ultra-short text (< 15 characters).")
print(f"remaining dataset size: {len(df):,}")


def parse_tags(tag_str: str):
    
    if not isinstance(tag_str, str):
        return []
    return [t.strip() for t in tag_str.split("|") if t.strip() != ""]


df["tags_list"] = df["question_tags"].apply(parse_tags)
df["tags_clean"] = df["tags_list"].apply(lambda tags: ", ".join(tags))
df["num_tags"] = df["tags_list"].apply(len)


df["answer_accepted"] = df["answer_accepted"].astype(bool)
df["target"] = df["answer_accepted"].astype(int)

print("\n--- Target Variable Distribution After Cleaning ---")
print(df["target"].value_counts(normalize=True).round(3))



df["question_length"] = df["question_body"].str.len()
df["answer_length"] = df["answer_body"].str.len()


final_columns = [
    "qid", "site", "url",
    "question_title", "question_body", "tags_clean", "num_tags",
    "question_score", "question_view_count", "question_answer_count",
    "answer_body", "answer_score", "answer_accepted", "target",
    "question_length", "answer_length",
]
df_clean = df[final_columns].reset_index(drop=True)


output_path = "cleaned_space_qa.csv"
df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\nCleaned dataset successfully saved to: {output_path}")
print(f"Final number of rows: {len(df_clean):,}")
print(f"Final number of columns: {df_clean.shape[1]}")
print("\n--- Preview of First 3 Rows ---")
print(df_clean.head(3))