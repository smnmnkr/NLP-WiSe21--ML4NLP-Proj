from typing import List, Dict

import re
import pandas as pd

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from transformers import logging

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.set_verbosity_warning()


def prepare_data(raw: pd.DataFrame) -> List[Dict]:
    def _clean_text(row: pd.DataFrame) -> pd.DataFrame:
        # remove hyperlinks
        # src: https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python/11332580
        row['text'] = re.sub(r'\S*https?:\S*', "", row['text'])

        # remove mentions
        row['text'] = re.sub(r'@\w*', "", row['text'])

        # remove hashtags
        row['text'] = re.sub(r'#\w*', "", row['text'])

        return row

    def _convert_sentiment(row: pd.DataFrame) -> pd.DataFrame:
        row['label'] = 0 if row['label'] == 'Neutral' else 1
        return row

    prepared = raw.copy()

    prepared.drop(['id', 'time', 'lang', 'smth'], axis=1, inplace=True)
    prepared.rename(columns={'tweet': 'text', 'sent': 'label'}, inplace=True)

    prepared.apply(_clean_text, axis=1)
    prepared.apply(_convert_sentiment, axis=1)

    return prepared.to_dict('records')


def tokenize(data: list, tokenizer):
    for row in data:
        row.update(tokenizer(row['text'], truncation=True, padding='max_length', max_length=500))
        del row['text']


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)

# Loading the data
train_raw = pd.read_csv('data/train.csv', sep=',')
test_raw = pd.read_csv('data/test.csv', sep=',')

train = prepare_data(train_raw)
test = prepare_data(test_raw)

tokenize(train, bert_tokenizer)
tokenize(test, bert_tokenizer)

bert_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train,
    eval_dataset=test,
    tokenizer=bert_tokenizer,
    data_collator=data_collator,

)

trainer.train()
trainer.evaluate()
