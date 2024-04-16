from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from transformers import AutoTokenizer as tokenizer

from huggingface_hub import login,create_repo
# from torch.utils.tensorboard import SummaryWriter
# http://127.0.0.1:6006/?darkMode=true#timeseries

login("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")

# 创建一个新的模型仓库
# create_repo(
#     repo_id="shixm-report-id",  # 仓库ID
#     token="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",  # 使用同样的token进行认证
#     # 更多配置...
# )
# torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 96.00 MiB. GPU 0 has a total capacity of 12.00 GiB of
# which 0 bytes is free. Of the allocated memory 26.51 GiB is allocated by PyTorch, and 44.25 MiB is reserved by PyTorch but
# unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to av
# oid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
#   0%|          | 0/10 [00:03<?, ?it/s]

num_labels = 6
model_ckpt ="distilbert/distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)
tokenizer = tokenizer.from_pretrained(model_ckpt)
def tokenize_function(examples):
    # return tokenizer(examples["text"], padding="max_length", truncation=True)
    return tokenizer(examples["text"],padding=True, truncation=True, max_length=20)
print("@@@@@@@@@@@@@@@@@@@@@@")
emotions = load_dataset("emotion")
emotions_train = emotions["train"]
# emotions_train = emotions_train.select(range(300))
# trainout = exec2(emotions_train)
trainout = emotions_train.map(tokenize_function, batched=True, batch_size=100)
print("@@@@@@@@@@@@@@@@@@@@@@")

emotions_validation = emotions["validation"]
# emotions_validation = emotions_validation.select(range(300))
# validationout = exec2(emotions_validation)
validationout = emotions_validation.map(tokenize_function, batched=True, batch_size=100)

print("@@@@@@@@@@@@@@@@@@@@@@")
print(f"train-ken:{len(trainout)},validation-len{len(validationout)}")
# showHex(validationout)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

batch_size = 64
logging_steps = len(emotions_train) // batch_size
print(f"logging_steps:{logging_steps}")
model_name =f"{model_ckpt}-finetuned-emotions-shixm"
train_args = TrainingArguments(output_dir=model_name,
                               num_train_epochs=5,
                               learning_rate=2e-5,
                               per_device_train_batch_size=batch_size,
                               per_device_eval_batch_size=batch_size,
                               weight_decay=0.001,
                               evaluation_strategy="epoch",
                               disable_tqdm=False,
                               logging_steps=logging_steps,
                               logging_strategy='steps',
                               push_to_hub=True,
                               log_level="error",
                               report_to=["tensorboard",],
                               logging_dir="logss",
                               )
trainer = Trainer(model=model, args=train_args, compute_metrics=compute_metrics, train_dataset=trainout, eval_dataset=validationout)
trainer.train()
# // 保存模型和tokenizer,以便以后推理加载的时候使用
tokenizer.save_pretrained(model_name)
