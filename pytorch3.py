import  torch
from transformers import  AutoModel
from transformers import AutoTokenizer as tokenizer

def print_type_structure(obj, indent=0):
    if isinstance(obj, (list, tuple)):
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


model_ckpt = AutoModel.from_pretrained("bert-base-uncased")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'GPU OR CPU: {device}')
model = AutoModel.from_pretrained("bert-base-uncased").to(device)
tokenizer = tokenizer.from_pretrained("bert-base-uncased")
text = "this is a text"
inputs = tokenizer(text, return_tensors="pt")
print(inputs)
print_type_structure(inputs)
print(inputs["input_ids"].size())
print(f'print format:{inputs["input_ids"].size()}')
inputs = {k:v.to(device) for k,v in inputs.items()}
print(inputs)
print_type_structure(inputs)

with torch.no_grad():
    outputs = model(**inputs)
print_type_structure(outputs)
print(outputs)