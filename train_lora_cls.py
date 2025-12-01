import os, numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

OUT_DIR = "distilbert-spam-lora"
MODEL_ID = "distilbert-base-uncased"

# Load dataset
ds = load_dataset("sms_spam")
cols = ds["train"].column_names
TEXT_COL = "sms" if "sms" in cols else ("text" if "text" in cols else cols[0])
LABEL_COL = "label" if "label" in cols else cols[-1]

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token if hasattr(tok, "eos_token") else tok.sep_token

def preprocess(batch):
    enc = tok(batch[TEXT_COL], truncation=True, padding="max_length", max_length=128)
    # normalize labels to 0/1
    lbls = []
    for v in batch[LABEL_COL]:
        if isinstance(v, str):
            s = v.strip().lower()
            lbls.append(1 if s == "spam" else 0)
        else:
            lbls.append(int(v))
    enc["labels"] = lbls
    return enc

ds_tok = ds.map(preprocess, batched=True, remove_columns=cols)
ds_tok.set_format(type="torch")

# Model + LoRA
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=2, id2label={0:"ham",1:"spam"}, label2id={"ham":0,"spam":1}
)
lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_lin","k_lin","v_lin","out_lin"]
)
model = get_peft_model(model, lora_cfg)

def metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1, zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

from train_lora_cls import *
# Minimal args that work everywhere; no evaluation_strategy/save_strategy/load_best_model_at_end
args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-4,
    weight_decay=0.01,
    logging_steps=25,
    report_to="none",
    do_eval=True  # allowed, but won't schedule eval during training
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["test"] if "test" in ds_tok else None,
    compute_metrics=metrics,
)

trainer.train()
# Manual eval after training since we didn't set evaluation_strategy
if "test" in ds_tok:
    print("[eval] test metrics:", trainer.evaluate(ds_tok["test"]))
trainer.save_model(OUT_DIR)
print(f"[done] Saved LoRA seq-cls model to {OUT_DIR}")
