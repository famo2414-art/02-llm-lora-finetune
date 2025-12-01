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
