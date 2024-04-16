import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoModelForSequenceClassification
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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


def exec3(emotions):
    model_ckpt = "distilbert/distilbert-base-uncased"
    model_name = f"{model_ckpt}-finetuned-emotions-shixm"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label_pre = []
    label_val = []
    for i in emotions:
        inputs = tokenizer(i["text"], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        max = torch.argmax(predictions,1)
        label_pre.append(max.item())
        label_val.append(i["label"])
    return label_pre,label_val

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
    map_label = {0: '悲伤', 1: '喜悦', 2: '爱', 3: '愤怒', 4: '恐惧', 5: '惊喜'}

    emotions = load_dataset("emotion")
    emotions_train = emotions["train"]
    emotions_train = emotions_train.select(range(2000))
    print("@@@@@@@@@@@@@@@@@@@@@@")
    trainout = exec3(emotions_train)
    print(f"train-ken:{len(trainout[0])},validation-len{len(trainout[1])}")
    accuracy = accuracy_score(trainout[0], trainout[1])
    print(f'train~~~Accuracy: {accuracy:.4f}')
    plot_confusion_matrix(trainout[0],trainout[1],map_label.values(),title="Confusion matrix (normalized)",cmap=plt.cm.Greens)

    # print_type_structure(trainout)
    # break run
    # showHex(trainout)
    print("@@@@@@@@@@@@@@@@@@@@@@")

    emotions_validation = emotions["validation"]
    emotions_validation = emotions_validation.select(range(2000))
    validationout = exec3(emotions_validation)
    print("@@@@@@@@@@@@@@@@@@@@@@")
    print(f"valid-ken:{len(validationout[0])},valid-len{len(validationout[1])}")
    accuracy = accuracy_score(validationout[0], validationout[1])
    print(f'valid***Accuracy: {accuracy:.4f}')
    plot_confusion_matrix(validationout[0],validationout[1],map_label.values(),title="Confusion matrix (normalized)",cmap=plt.cm.Greens)


if __name__ == '__main__':
    main()
    # requests.exceptions.SSLError: (MaxRetryError(
    #     "HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /bert-base-uncased/resolve/main/config.json (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1006)')))"),
    #                                '(Request ID: 3b533093-dbe1-4861-87fc-04bda776d007)')
