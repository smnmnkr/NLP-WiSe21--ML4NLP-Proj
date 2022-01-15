import pandas as pd
from typing import List, Dict
import re

from transformers import AutoTokenizer

import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, SequentialSampler


def _load_dataset(data_path):
    return pd.read_csv(data_path, sep=',')

def _prepare_data(raw: pd.DataFrame, custom_clean_text=None) -> List[Dict]:
    def _clean_text(row: pd.DataFrame) -> pd.DataFrame:
        # remove hyperlinks
        # src: https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python/11332580
        row['text'] = re.sub(r'\S*https?:\S*', "", row['text'])

        # remove mentions
        row['text'] = re.sub(r'@\w*', "", row['text'])

        # remove hashtags
        row['text'] = re.sub(r'#\w*', "", row['text'])

        # remove emojis
        pattern = re.compile("["
                             u"\U0001F600-\U0001F64F"  # emoticons
                             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                             u"\U0001F680-\U0001F6FF"  # transport & map symbols
                             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                             "]+", flags=re.UNICODE)

        row['text'] = pattern.sub(r' ', row['text'])

        return row

    def _convert_sentiment(row: pd.DataFrame) -> pd.DataFrame:
        row['label'] = 0 if row['label'] == 'Neutral' else 1
        return row

    prepared = raw.copy()

    prepared.drop(['id', 'time', 'lang', 'smth'], axis=1, inplace=True)
    prepared.rename(columns={'tweet': 'text', 'sent': 'label'}, inplace=True)

    if custom_clean_text:
        prepared.apply(custom_clean_text, axis=1)
    else:
        prepared.apply(_clean_text, axis=1)
    
    prepared.apply(_convert_sentiment, axis=1)

    return prepared

def _create_tensors(df, model_name):
    sentences = df.text.values
    labels = df.label.values.astype(int)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer(
                            sent,
                            truncation=True,
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]', True by default
                            max_length = 128,           # Pad & truncate all sentences.
                            padding='max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
    
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return input_ids, attention_masks, labels

def _create_dataloaders(input_ids, attention_masks, labels, batch_size, create_validation_set):

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    if create_validation_set:
        # Create a 90-10 train-validation split.
        # Calculate the number of samples to include in each set.
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        dataset, val_dataset = random_split(dataset, [train_size, val_size])

        validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )

    dataloader = DataLoader(
                dataset,  # The training samples.
                shuffle = False,
                batch_size = batch_size # Trains with this batch size.
            )

    if create_validation_set:
        return dataloader, validation_dataloader
    else:
        return dataloader

def get_dataloaders(data_path,  model_name = "bert-base-cased", batch_size = 16, create_validation_set = True):
    df = _load_dataset(data_path)
    df = _prepare_data(df)
    input_ids, attention_masks, labels = _create_tensors(df, model_name)
    dataloaders = _create_dataloaders(input_ids, attention_masks, labels, batch_size, create_validation_set)

    return dataloaders