import numpy
import  torch
from datasets import load_dataset
from transformers import  AutoModel
from transformers import AutoTokenizer as tokenizer

def print_type_structure(obj, indent=0):
    if isinstance(obj, numpy.ndarray):
        print(' ' * indent + f'ndarray.size(): {obj.size}')
    elif  isinstance(obj, torch.Tensor):
        print(' ' * indent + f'Tensor: {obj.size()}')
    elif isinstance(obj, (list, tuple)):
        print(' ' * indent + 'List/Tuple:')
        for item in obj:
            print_type_structure(item, indent + 4)
    elif isinstance(obj, dict):
        print(' ' * indent + 'Dictionary:')
        for key, value in obj.items():
            print(' ' * (indent + 4) + f'Key: {type(key).__name__} # ->"'+key+'", Value:')
            print_type_structure(value, indent + 8)
    else:
        print(' ' * indent + f'Type: {type(obj).__name__}')
def export_hidden_states(batch):
    # inputs = {k:v.to(device)for k,v in batch.items()}
    inputs = {k: v for k, v in batch.items()}
    with torch.no_grad():
        last_hidden_states = model(**inputs).last_hidden_state
    return {"hidden_state":last_hidden_states[:,0].cpu().numpy()}
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)



model_ckpt = AutoModel.from_pretrained("bert-base-uncased")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'GPU OR CPU: {device}')
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = tokenizer.from_pretrained("bert-base-uncased")
text = "this is a text 我操！"
inputs = tokenizer(text, return_tensors="pt")
print(inputs)
print_type_structure(inputs)
print(inputs["input_ids"].size())
print(f'print format:{inputs["input_ids"].size()}')
inputs = {k:v for k,v in inputs.items()}
print(inputs)
print_type_structure(inputs)

with torch.no_grad():
    outputs = model(**inputs)
print_type_structure(outputs)
# print(outputs)
print(outputs.last_hidden_state.size())
# print(outputs.last_hidden_state)
ret_map = {"ret_hidden_map":outputs.last_hidden_state[:,0].cpu().numpy()}
print_type_structure(ret_map)
# print(ret_map)
print("@@@@@@@@@@@@@@@@@@@@@@")
# 加载情感数据集
emotions = load_dataset("emotion")
emotions = emotions["train"]
emotions = emotions.select(range(1000))

# emotions.set_format("panda")
# print_type_structure(emotions)
# tokenizer(emotions["train"][:]["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
# print_type_structure(emotions)
emotions_encoded=emotions.map(tokenize_function, batched=True, batch_size=None)
print_type_structure(emotions_encoded)
print("^^^^^^^^^^^^")
emotions_encoded.set_format(type="torch", columns=["attention_mask","input_ids", "label"])
print_type_structure(emotions_encoded)

# emotions_encoded = emotions_encoded.select(["attention_mask","input_ids", "label"])
emotions_encoded=emotions_encoded.remove_columns(["text"])
emotions_encoded=emotions_encoded.remove_columns(["label"])

emotions_encoded = emotions_encoded.map(export_hidden_states, batched=True)
print_type_structure(emotions_encoded)
print(emotions_encoded.column_names)