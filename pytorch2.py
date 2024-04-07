import unittest
from io import StringIO
import sys
import pandas as pd
pd.set_option('display.max_columns', None)

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


def train():
    """
    加载和处理情感数据集，并使用GPT-2模型进行训练。
    该函数不接受参数，也不返回任何值。
    """
    from datasets import load_dataset
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
    import matplotlib.pyplot as plt
    import pandas as pd

    # 加载情感数据集
    emotions = load_dataset("emotion")
    print(emotions["train"].features)
    print(type(emotions))
    print_type_structure(emotions)

    # 打印数据集的长度
    print(len(emotions["train"]))
    # class 'datasets.dataset_dict.DatasetDict'>
    # class 'datasets.arrow_dataset.Dataset'>
    #  dataset -> [:] -> dataframe
    # class 'pandas.core.frame.DataFrame'>
    # 将数据集格式设置为pandas，以便于使用DataFrame进行操作
    emotions.set_format("pandas")

    # class 'pandas.core.frame.DataFrame'>

    # 在Python中，当你看到dataset[:]
    # 这样的表达式，它通常表示对名为dataset的序列（如列表、元组、字符串或任何实现了__getitem__和__len__方法的对象）进行切片操作。这里的切片操作[
    #                                                                                                                      :]
    # 实际上创建了一个dataset的浅拷贝。
    #
    # 对于大多数常见的序列类型，dataset[:]
    # 将返回一个新序列，该序列包含dataset中的所有元素，但它是独立于原始dataset的新对象。对dataset[:]
    # 所做的任何修改都不会影响原始的dataset


    df = emotions["train"][:]
    print("emotions-train-type:" +str(type(emotions["train"])))

    print("df-type:" +str(type(df)))
    print(df.head())
    # 打印测试集的前5条数据
    print(emotions["test"][0:5])


    # class 'datasets.features.features.ClassLabel'>
    print(type(emotions["train"].features["label"]))
    # 将标签转换为可读的字符串形式
    # DataFrame.apply()
    # 方法用于沿着轴的方向应用一个函数。默认情况下，apply()
    # 方法不会修改原始DataFrame，而是返回一个新的对象。
    df["label_name"] = df["label"].apply(lambda x: emotions["train"].features["label"].int2str(x))
    # 统计并绘制不同情感标签的数量分布
    df["label_name"].value_counts().plot(kind="bar")
    print(df.head())
    emotions.map(tokenizer,batched=True,batch_size=None)
    tokenizer()
    # 显示情感分布图
    plt.title("FIVE EMOTIONS")
    # plt.show()
def testDataFrameFeture ():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    print_type_structure(df)

from datasets import Dataset
import numpy as np
def generate_random_data(num_samples, num_features):
    # 生成指定数量的样本和特征
    data = np.random.rand(num_samples, num_features)
    # 添加标签（例如，一个简单的二分类问题）
    labels = np.random.randint(0, 2, size=num_samples)
    # 将数据和标签组合成字典列表
    samples = [{'data': list(row), 'label': label} for row, label in zip(data, labels)]
    return samples

from transformers import  DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def tokenize(batch):
    data = batch["train"][:5]["text"]
    print_type_structure(data)
    return tokenizer(data, padding=True, truncation=True)

def train2():

    text = "hello everyone 我是一个中国人"
    split_w1 = tokenizer.tokenize(text)
    print(tokenizer.convert_tokens_to_string(split_w1))
    print(split_w1)
    print(tokenizer.vocab_size)
    print(tokenizer.model_max_length)
    print(tokenizer.encode(text))
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))
    print(tokenizer(text,padding=True,truncation=True))
def train3():
    from datasets import load_dataset
    # 加载情感数据集
    emotions = load_dataset("emotion")
    emotions.set_format("pandas")
    df = emotions["train"][:]
    print(df[:5])
    df["label_name"] =df["label"].apply(lambda x: emotions["train"].features["label"].int2str(x))
    df["label_str"] = df["label"].apply(lambda x: str(x))
    print(df[:5])
    emotions.reset_format()
    print(tokenize(emotions))

# def tokenize(batch):
#     pass
#     # return tokenizer.

if __name__ == '__main__':
    ds = generate_random_data(2,3)
    dss = Dataset.from_list(ds)
    print(type(dss))
    print(ds)
    # print(dss.features["label"].int2str(0))
    testDataFrameFeture()
    train3()