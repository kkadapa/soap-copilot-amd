import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

MODEL = "/workspace/models/llama3-8b"
DATA = "/workspace/data/soap_dataset.jsonl"
OUTPUT = "/workspace/models/soap_model"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model in bf16...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

def format_example(ex):
    return {"text": (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{ex['instruction']}\n\n{ex['input']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{ex['output']}<|eot_id|>"
    )}

print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA, split="train")
dataset = dataset.map(format_example)
print(f"Dataset size: {len(dataset)} examples")

cfg = SFTConfig(
    output_dir=OUTPUT,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=5,
    save_steps=50,
    bf16=True,
    max_seq_length=2048,
    dataset_text_field="text",
    gradient_checkpointing=True,
    report_to="none",
)

print("Starting training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=cfg,
    tokenizer=tokenizer,
)

trainer.train()

print("Saving model...")
trainer.model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)
print(f"Model saved to {OUTPUT}")
