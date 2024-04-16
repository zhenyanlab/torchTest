import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import  torch
from datasets import load_dataset
from transformers import  AutoModel
from transformers import AutoTokenizer as tokenizer


import umap
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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


# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument index in method wrapper_CUDA__index_select)

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
emotions = emotions.select(range(450))

# emotions.set_format("panda")
# print_type_structure(emotions)
# tokenizer(emotions["train"][:]["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
# print_type_structure(emotions)
emotions_encoded = emotions.map(tokenize_function, batched=True, batch_size=None)
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


# pip install umap-learn
# pip install scikit-learn
from matplotlib import rcParams

# 设置字体为支持中文的字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

x_train = np.array(emotions_encoded["hidden_state"])
x_scaled = MinMaxScaler().fit_transform(x_train)
xumap = umap.UMAP(n_components=2).fit(x_scaled)
df_train_umap = pd.DataFrame(xumap.embedding_, columns=["x", "y"])
df_train_umap["label"] = np.array(emotions["label"])
print(df_train_umap.head(10))
map_label = {0:'悲伤',1:'喜悦',2:'爱',3:'愤怒',4:'恐惧',5:'惊喜'}
cmaps = ["Greens","Blues","Reds","Purples","Oranges","Greys"]
fig,axes = plt.subplots(2,3,figsize=(10,10))
axes = axes.flatten()
print(emotions.features["label"])

for i, (label,cmap) in enumerate(zip(df_train_umap["label"].unique(),cmaps)):
    print(f"label == {label},i={i},axes={axes}")
    df_train_umap_label = df_train_umap.query(f"label == {label}")
    axes[i].hexbin(df_train_umap_label["x"], df_train_umap_label["y"], label=label,gridsize=20,linewidths=(0,),cmap=cmap)
    axes[i].set_title(map_label[label])

plt.show()
