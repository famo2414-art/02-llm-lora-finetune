import os, math, gradio as gr

LOLA_DIR = "distilbert-spam-lora"
HAS_LORA = os.path.isdir(LOLA_DIR) and any(f.endswith((".bin",".safetensors")) for f in os.listdir(LOLA_DIR))

clf = None
mode = None

if HAS_LORA:
    from transformers import pipeline, AutoConfig
    cfg = AutoConfig.from_pretrained(LOLA_DIR)
    id2label = getattr(cfg, "id2label", {0:"LABEL_0", 1:"LABEL_1"})
    clf = pipeline("text-classification", model=LOLA_DIR)
    mode = "lora"
else:
    import joblib
    if os.path.exists("spam_clf.joblib"):
        sk = joblib.load("spam_clf.joblib")  # 0=ham, 1=spam
        mode = "sklearn"
    else:
        raise SystemExit("No LoRA model and no baseline 'spam_clf.joblib' found. Train one of them first.")

def sigmoid(x): return 1/(1+math.exp(-x))

def predict(text: str):
    text = (text or "").strip()
    if not text:
        return {"Verdict": "not spam", "Confidence": 0.5}

    if mode == "lora":
        out = clf(text)[0]  # {'label': 'ham'/'spam' or 'LABEL_0/1', 'score': float}
        lbl = out["label"].lower()
        if "spam" in lbl or lbl.endswith("1"):
            return {"Verdict": "spam", "Confidence": round(float(out.get("score", 0.5)), 3)}
        return {"Verdict": "not spam", "Confidence": round(float(out.get("score", 0.5)), 3)}
    else:
        if hasattr(sk, "decision_function"):
            d = float(sk.decision_function([text])[0])
            prob = sigmoid(d)
            pred = int(d > 0)
        else:
            prob = float(sk.predict_proba([text])[0][1])
            pred = int(prob >= 0.5)
        return {"Verdict": "spam" if pred==1 else "not spam", "Confidence": round(prob, 3)}

gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Message", placeholder="Win a free iPhone now!"),
    outputs=gr.JSON(label="Result"),
    title=("SMS Spam Detector — LoRA DistilBERT" if mode=="lora" else "SMS Spam Detector — TF-IDF + LinearSVC"),
    description="Binary classifier with confidence; uses LoRA model if available, otherwise a fast TF-IDF baseline."
).launch()
