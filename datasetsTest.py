from datasets import  list_datasets
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import  datasets

# Show ALL THE DATASETS
#print(list_datasets()[:10])
emotions = load_dataset("emotion")
print(emotions)
print(len(emotions))
print(len(emotions["train"]))
print(emotions["train"].features)
print(emotions["train"]["label"][:5])
print("_______________________")
emotions.set_format(type="pandas")
df = emotions["train"][:]
df.head()
print("_______________________")

def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

df = emotions["train"].to_pandas()

df["label_name"] = df["label"].apply(label_int2str)
df.head()
print("_______________________")
df["label_name"].value_counts(ascending=True).plot(kind="barh")
plt.title("Frequency of Classes")
plt.show()