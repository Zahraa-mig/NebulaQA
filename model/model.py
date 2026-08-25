
import torch
import pandas as pd

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig



print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ ما في GPU! روحي Runtime -> Change runtime type -> اختاري T4 GPU")



dataset = load_dataset("juliensimon/stackexchange-space-qa", split="train")
print(dataset)

df = dataset.to_pandas()


df = df[
    (df["answer_accepted"] == True)
    & (df["question_score"] >= 5)
    & (df["answer_body"].notna())
]
print(f"عدد الأمثلة بعد الفلترة: {len(df):,}")
print(df.head())


sft = {
    "prompt": (
        df["question_title"].fillna("") + "\n\n" + df["question_body"].fillna("")
    ).tolist(),
    "completion": df["answer_body"].fillna("").tolist(),
}
sft_dataset = Dataset.from_dict(sft)
print(sft_dataset)



split_dataset = sft_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
temp_dataset = split_dataset["test"]

split_temp = temp_dataset.train_test_split(test_size=0.5, seed=42)
valid_dataset = split_temp["train"]
test_dataset = split_temp["test"]

print("Train:", len(train_dataset))
print("Valid:", len(valid_dataset))
print("Test:", len(test_dataset))




model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token



def format_example(example):
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return example

train_dataset = train_dataset.map(format_example)
valid_dataset = valid_dataset.map(format_example)

train_dataset = train_dataset.remove_columns(["prompt", "completion"])
valid_dataset = valid_dataset.remove_columns(["prompt", "completion"])

print("أعمدة train_dataset بعد التنظيف:", train_dataset.column_names)
print("عدد صفوف train_dataset:", len(train_dataset))
print("عدد صفوف valid_dataset:", len(valid_dataset))



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.config.pad_token_id = tokenizer.pad_token_id

model = prepare_model_for_kbit_training(model)


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)


training_args = SFTConfig(
    output_dir="./qwen-qlora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    dataset_text_field="text",
    report_to="none",
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)


def print_trainable_parameters(m):
    trainable_params = 0
    all_param = 0
    for _, param in m.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )


print_trainable_parameters(trainer.model)

trainer.train()

trainer.save_model("./qwen-qlora-final")
tokenizer.save_pretrained("./qwen-qlora-final")

eval_results = trainer.evaluate()
print(eval_results)


question = "Why does the Moon always show nearly the same face to Earth?"

model = trainer.model
model.eval()

messages = [{"role": "user", "content": question}]
prompt_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = tokenizer(prompt_text, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
    )

input_length = inputs["input_ids"].shape[1]
generated_tokens = outputs[0][input_length:]
answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("Question:")
print(question)
print("\nModel answer:")
print(answer)
