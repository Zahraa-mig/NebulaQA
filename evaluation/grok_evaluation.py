import os
import json
import pandas as pd
from openai import OpenAI


XAI_API_KEY = "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"

client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")


INPUT_CSV = "evaluation_results.csv"
N_QUESTIONS = 15 

df = pd.read_csv(INPUT_CSV)
sample = df.sample(min(N_QUESTIONS, len(df)), random_state=42).reset_index(drop=True)

SYSTEM_PROMPT = """You are an objective AI evaluator judging a fine-tuned model's answers
to space/astronomy questions against a reference (expert) answer.

Score the model answer from 1 to 5 on each criterion:
- correctness: does it contain the same facts/logic as the reference answer?
- completeness: did it cover the important details from the reference?
- hallucination: 5 = no invented info, 1 = mostly made up or contradicts the reference.

Respond ONLY with valid JSON, no extra text, in this exact format:
{"correctness": <int>, "completeness": <int>, "hallucination": <int>, "reason": "<short reason>"}
"""

def judge(question, reference_answer, model_answer):
    user_prompt = f"""Question: {question}

Reference (expert) answer: {reference_answer}

Model answer: {model_answer}
"""
    response = client.chat.completions.create(
        model="grok-4.3",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

results = []
for i, row in sample.iterrows():
    try:
        scores = judge(row["question"], row["reference_answer"], row["generated_answer"])
    except Exception as e:
        scores = {"correctness": None, "completeness": None, "hallucination": None, "reason": f"error: {e}"}

    results.append({
        "question": row["question"],
        "reference_answer": row["reference_answer"],
        "generated_answer": row["generated_answer"],
        "correctness": scores.get("correctness"),
        "completeness": scores.get("completeness"),
        "hallucination": scores.get("hallucination"),
        "reason": scores.get("reason"),
    })
    print(f"{i + 1}/{len(sample)} done")

results_df = pd.DataFrame(results)
results_df.to_csv("grok_evaluation_results.csv", index=False, encoding="utf-8-sig")

print("\n--- averages ---")
print("correctness:", results_df["correctness"].mean())
print("completeness:", results_df["completeness"].mean())
print("hallucination:", results_df["hallucination"].mean())
print("\nsaved: grok_evaluation_results.csv")
