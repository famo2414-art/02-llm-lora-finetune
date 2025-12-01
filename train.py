from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

MODEL = "microsoft/DialoGPT-small"

# tokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

# data
ds = load_dataset("sms_spam")
def preprocess(batch):
    texts = ["Is this spam? " + t for t in batch["text"]]
    labels = [" spam" if int(l)==1 else " not spam" for l in batch["label"]]
    enc = tok(texts, truncation=True, padding="max_length", max_length=128)
    lab = tok(labels, truncation=True, padding="max_length", max_length=4)
    enc["labels"] = lab["input_ids"]
    return enc

tokd = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

# model + LoRA
m = AutoModelForCausalLM.from_pretrained(MODEL)
lcfg = LoraConfig(
    r=16, lora_alpha=32, target_modules=["c_attn"], lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM"
)
m = get_peft_model(m, lcfg)

# training (short run; bump epochs later)
args = TrainingArguments(
    output_dir="dialo-gpt-spam-lora",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=20,
    save_steps=200,
    report_to="none",
)

trainer = Trainer(model=m, args=args, train_dataset=tokd["train"])
trainer.train()
trainer.save_model("dialo-gpt-spam-lora")
