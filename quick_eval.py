import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report

OUT = "distilbert-spam-lora"
tok = AutoTokenizer.from_pretrained(OUT)
model = AutoModelForSequenceClassification.from_pretrained(OUT)

ds = load_dataset("sms_spam")
test = ds["test"]

# batch encode to avoid GPU/MPS warnings; pure CPU is fine
texts = list(test["sms"])
labels = np.array(test["label"])

enc = tok(texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
with np.no_grad():
    logits = model(**{k: v for k, v in enc.items()}).logits.detach().cpu().numpy()
preds = logits.argmax(axis=-1)

acc = (preds == labels).mean()
print(f"Accuracy: {acc:.4f}")
print(classification_report(labels, preds, target_names=["ham","spam"]))
