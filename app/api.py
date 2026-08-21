import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_PATH = "../model/qwen-qlora-final"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

app = FastAPI(title="NebulaQA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


class Question(BaseModel):
    question: str
    max_new_tokens: int = 150


class Answer(BaseModel):
    question: str
    answer: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(payload: Question):
    messages = [{"role": "user", "content": payload.question}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=payload.max_new_tokens,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    answer_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return Answer(question=payload.question, answer=answer_text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
