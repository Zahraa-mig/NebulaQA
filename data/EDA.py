import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")
OUTPUT_DIR = "eda_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig_name):
    path = os.path.join(OUTPUT_DIR, fig_name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


df = pd.read_csv("cleaned_space_qa.csv")
print(f"Number of rows: {len(df):,} | Number of columns: {df.shape[1]}\n")


print("--- Descriptive Statistics for Numerical Columns ---")
print(df[["question_score", "answer_score", "question_view_count",
          "question_answer_count", "question_length", "answer_length"]].describe().round(1))



print("\n 1) Target Distribution Analysis")
plt.figure(figsize=(5, 4))
counts = df["target"].value_counts().sort_index()
sns.barplot(
    x=["Not Accepted (0)", "Accepted (1)"],
    y=counts.values,
    palette=["#e57373", "#64b5f6"]
)
plt.title("Distribution of Accepted vs. Not Accepted Answers")
plt.xlabel("Answer Acceptance Status")
plt.ylabel("Count of Answers")

save("01_target_distribution.png")



print("2) Question Distribution by Source Site")
plt.figure(figsize=(5, 4))
site_counts = df["site"].value_counts()
sns.barplot(
    x=site_counts.index, 
    y=site_counts.values, 
    palette="crest"
)
plt.title("Number of Questions by Source (Astronomy vs. Space)")
plt.xlabel("Source Site")
plt.ylabel("Number of Questions")

save("02_site_distribution.png")



print("3) Score Distribution for Questions and Answers")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(df["question_score"].clip(-5, 50), bins=40, ax=axes[0], color="#64b5f6")
axes[0].set_title("Question Score Distribution")
axes[0].set_xlabel("Question Score")
axes[0].set_ylabel("Count")
sns.histplot(df["answer_score"].clip(-5, 50), bins=40, ax=axes[1], color="#81c784")
axes[1].set_title("Answer Score Distribution")
axes[1].set_xlabel("Answer Score")
axes[1].set_ylabel("Count")

plt.tight_layout()
save("03_score_distributions.png")



print("4) Question Length vs. Answer Length Relationship")
plt.figure(figsize=(6, 5))
sample = df.sample(min(3000, len(df)), random_state=42)  # Sample for faster & clearer visualization

sns.scatterplot(
    data=sample, 
    x="question_length", 
    y="answer_length",
    hue="target", 
    alpha=0.4, 
    palette=["#e57373", "#64b5f6"]
)

plt.title("Question Length vs. Answer Length")
plt.xlabel("Question Length (characters)")
plt.ylabel("Answer Length (characters)")
save("04_length_relationship.png")



print("5) Top 15 Most Frequent Tags")
all_tags = df["tags_clean"].str.split(", ").explode()
top_tags = all_tags.value_counts().head(15)

plt.figure(figsize=(7, 6))
sns.barplot(x=top_tags.values, y=top_tags.index, palette="flare")

plt.title("Top 15 Most Frequent Tags in Questions")
plt.xlabel("Number of Questions")
plt.ylabel("Tag")
save("05_top_tags.png")



print("6) Correlation Matrix of Numeric Features")
numeric_cols = [
    "question_score", "answer_score", "question_view_count",
    "question_answer_count", "question_length", "answer_length", "target"
]

plt.figure(figsize=(7, 6))
corr = df[numeric_cols].corr().round(2)
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Numeric Features")
save("06_correlation_heatmap.png")


print("7) Acceptance Rate by Number of Tags")
plt.figure(figsize=(6, 4))
acceptance_by_tags = df.groupby("num_tags")["target"].mean()

sns.lineplot(x=acceptance_by_tags.index, y=acceptance_by_tags.values, marker="o")
plt.title("Accepted Answer Rate by Number of Tags")
plt.xlabel("Number of Tags")
plt.ylabel("Acceptance Rate")
save("07_acceptance_by_tags.png")

print(f"\nAnalysis Complete! All figures saved in: {OUTPUT_DIR}/")
