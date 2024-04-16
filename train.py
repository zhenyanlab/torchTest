import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import  AutoModel
from transformers import AutoTokenizer


import umap
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from matplotlib import rcParams
# 设置字体为支持中文的字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# pip install umap-learn
# pip install scikit-learn

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

def print_type_structure(obj, indent=0):
    if isinstance(obj, np.ndarray):
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



def exec2(emotions):
    emotions_encoded = emotions.map(tokenize_function, batched=True, batch_size=100)
    print_type_structure(emotions_encoded)
    print("^^^^^^^^^^^^")
    emotions_encoded.set_format(type="torch", columns=["attention_mask","input_ids", "label"])
    print_type_structure(emotions_encoded)

    # emotions_encoded = emotions_encoded.select(["attention_mask","input_ids", "label"])
    emotions_encoded=emotions_encoded.remove_columns(["text"])
    tempLabel = emotions_encoded["label"]
    emotions_encoded=emotions_encoded.remove_columns(["label"])

    emotions_encoded = emotions_encoded.map(export_hidden_states, batched=True, batch_size=100)
    emotions_encoded = emotions_encoded.add_column("label", emotions["label"])

    print_type_structure(emotions_encoded)
    print("emotions_encoded:" + str(emotions_encoded.column_names))
    # emotions_encoded.add_column("label", tempLabel)

    return emotions_encoded

def showHex(emotions_encoded):
    x_train = np.array(emotions_encoded["hidden_state"])
    x_scaled = MinMaxScaler().fit_transform(x_train)
    xumap = umap.UMAP(n_components=2).fit(x_scaled)
    df_train_umap = pd.DataFrame(xumap.embedding_, columns=["x", "y"])
    df_train_umap["label"] = np.array(emotions_encoded["label"])
    print(df_train_umap.head(10))
    map_label = {0: '悲伤', 1: '喜悦', 2: '爱', 3: '愤怒', 4: '恐惧', 5: '惊喜'}
    cmaps = ["Greens", "Blues", "Reds", "Purples", "Oranges", "Greys"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 10))
    axes = axes.flatten()
    print(emotions_encoded.features["label"])
    for i, (label, cmap) in enumerate(zip(df_train_umap["label"].unique(), cmaps)):
        print(f"label == {label},i={i},axes={axes}")
        df_train_umap_label = df_train_umap.query(f"label == {label}")
        axes[i].hexbin(df_train_umap_label["x"], df_train_umap_label["y"], label=label, gridsize=20,
                       linewidths=(0,), cmap=cmap)
        axes[i].set_title(map_label[label])
    plt.show()


def plot_confusion_matrix(y_preds,y_true,labels,title='Confusion matrix', cmap=plt.cm.Blues):
    from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_preds,normalize="true")
    fig,ax = plt.subplots(figsize=(8,8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    disp.plot(include_values=True,cmap=cmap, ax=ax,values_format='.2f',xticks_rotation=45)
    plt.title(title)
    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'GPU OR CPU: {device}')

    emotions = load_dataset("emotion")
    emotions_train = emotions["train"]
    emotions_train = emotions_train.select(range(300))
    print("@@@@@@@@@@@@@@@@@@@@@@")
    trainout = exec2(emotions_train)
    # break run
    # showHex(trainout)
    print("@@@@@@@@@@@@@@@@@@@@@@")

    emotions_validation = emotions["validation"]
    emotions_validation = emotions_validation.select(range(300))
    validationout = exec2(emotions_validation)
    # break run
    # showHex(validationout)

    x_train= np.array(trainout["hidden_state"])
    x_val = np.array(validationout["hidden_state"])
    y_train = np.array(trainout["label"])
    y_val = np.array(validationout["label"])
    print(x_train.shape,x_val.shape,y_train.shape,y_val.shape)
    lr_clf = LogisticRegression()
    lr_clf.fit(x_train,y_train)
    print(lr_clf.score(x_val,y_val))

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(x_train,y_train)
    print(dummy_clf.score(x_val,y_val))
    map_label = {0: '悲伤', 1: '喜悦', 2: '爱', 3: '愤怒', 4: '恐惧', 5: '惊喜'}
    y_preds = lr_clf.predict(x_val)
    plot_confusion_matrix(y_preds,y_val,map_label.values(),title="Confusion matrix (normalized)",cmap=plt.cm.Greens)

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
if __name__ == '__main__':

    main()
    # requests.exceptions.SSLError: (MaxRetryError(
    #     "HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /bert-base-uncased/resolve/main/config.json (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1006)')))"),
    #                                '(Request ID: 3b533093-dbe1-4861-87fc-04bda776d007)')
