import pandas as pd
from model_preparation import set_seed
from data_preparation import _load_dataset, _prepare_data
import nltk
import subprocess
import os

"""
Reference:
https://github.com/jasonwei20/eda_nlp
"""
nltk.download('wordnet')
nltk.download('omw-1.4')



data_path = "data/train.csv"
model_name = 'microsoft/deberta-base'
batch_size = 16
create_validation_set = False
SEED = 42

set_seed(SEED)


original_df = _load_dataset(data_path)

original_df = _prepare_data(original_df)

original_df.rename(columns={'text': 'sentence'}, inplace=True)
original_df = original_df[["label", "sentence"]]
original_df.to_csv("BERT_in_Pytorch/eda_nlp-master/data/original_train.txt", index=False, header=False, sep="\t")


result = subprocess.run(["python","BERT_in_Pytorch/eda_nlp-master/code/augment.py","--input=original_train.txt","--output=augmented_train.txt"],
                        stdout=subprocess.PIPE, shell=True)
print(result)


augmented_df = pd.read_csv("augmented_train.txt", names = ["label", "text"], sep = "\t")
augmented_df = augmented_df.sample(frac=1).reset_index(drop=True)
augmented_df.to_csv("data/train_aug.csv", index=False)

os.remove("BERT_in_Pytorch/eda_nlp-master/data/original_train.txt")
os.remove("augmented_train.txt")