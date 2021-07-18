import random
from typing import Optional

import pytorch_lightning as pl
import torch
import pickle as pkl
import pandas as pd
import h5py
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, IterableDataset, random_split
import re
from transformers import AutoModel, BertTokenizer

from torch.utils import data


class IterCorpus:
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for line in open(self.dirname, 'r', encoding='utf-8'):
            yield line.split()


class BertTextDataset(data.Dataset):
    def __init__(self, text):
        self.text = text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError

        return self.text[idx]

class BaseDataset:
    def __init__(self, dir, feature_cols, file_type='csv', normalized=0, label_cols=None, chunk_size=None):
        self.dir = dir
        self.normalized = normalized

        with open(dir, 'r') as f:
            if file_type == 'csv':
                self.df = pd.read_csv(dir)
            if file_type == 'json':  # For the romanian sentiment task
                self.df = pd.json_normalize(pd.read_json(dir)['reviews'])

        if self.normalized == 0:
            self.normalize_df()

        self.feature_cols = feature_cols
        self.label_cols = label_cols if label_cols else None

    def normalize_df(self):
        equal_to_label = ["starRating"]
        for text in equal_to_label:
            if text in self.df.columns.tolist():
                self.df.rename(columns={text: 'label'}, inplace=True)

        equal_to_text = ['content', 'review']
        for text in equal_to_text:
            if text in self.df.columns.tolist():
                self.df.rename(columns={text: 'text'}, inplace=True)

    def get_features(self):
        return self.df.loc[:, self.feature_cols]

    def get_labels(self, label_cols='label'):
        return self.df.loc[:, label_cols]

    def gender_swap(self):
        pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError

        item = {"text": [self.df.loc[idx, feature] for feature in self.feature_cols],
                "labels": [self.df.loc[idx, label] for label in self.label_cols]}
        return item



class BertDataset(torch.utils.data.Dataset):
    def __init__(self, text, labels, tokenizer, model, cached=False, cache_name=None):

        self.tokenizer = tokenizer
        self.model = model
        self.text = text
        self.labels = labels

        if cached:
            self.encodings = pkl.load(open(cache_name, 'rb'))
        else:
            self.encodings = tokenizer.batch_encode_plus(text,
                                                         truncation=True,
                                                         padding=True,
                                                         max_length=120, return_tensors='pt')['input_ids']
        self.data = TensorDataset(self.encodings, self.labels)

        if cache_name and not cached:
            cache_file(cache_name, self.encodings)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError

        return self.encodings[idx]

    def get_loader(self, batch_size=64):
        sampler = RandomSampler(self.data)
        dataloader = DataLoader(self.data, sampler=sampler, batch_size=batch_size)
        return dataloader

    def split(self, ratios=None):
        size = len(self)
        if ratios is None:
            ratios = list(map(int, [0.8 * size, 0.1 * size, 0.1 * size]))

        train_set, val_set, test_set = data.random_split(self.data,ratios)
        return train_set, val_set, test_set


class BertMasked(BertDataset):
    def __init__(self, text, tokenizer, model, cached=False, cache_name=None):
        super().__init__(text, torch.zeros(text.values.shape), tokenizer, model, cached, cache_name)

        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        self.mask_encoding, self.labels, self.weights = self.mask()
        self.data = TensorDataset(self.mask_encoding, self.labels, self.weights)

    def __str__(self):
        s = ""
        for tokens in self.mask_encoding:
            sent = re.sub('(\[PAD\])*', '', self.tokenizer.decode(tokens))
            s += sent + '\n'
        return s

    def mask(self):
        # 15% BERT masking
        inp_mask = (torch.rand(self.encodings.size()) < 0.15)
        inp_mask[self.encodings <= 2] = False

        labels = -1 * torch.ones_like(self.encodings)  # Set default ignore
        labels[inp_mask] = self.encodings[inp_mask]
        encodings_masked = torch.clone(self.encodings)
        # Set input to [MASK]
        inp_mask_2mask = inp_mask & (torch.rand(self.encodings.size()) < 0.90)
        encodings_masked[
            inp_mask_2mask
        ] = self.mask_token_id

        # Set 10% to a random token
        inp_mask_2random = inp_mask_2mask & (torch.rand(self.encodings.size()) < 1 / 9)
        encodings_masked[inp_mask_2random] = torch.randint(
            self.mask_token_id, self.vocab_size, inp_mask_2random.type(torch.uint8).sum().shape  # CHECK!!
        )

        sample_weights = torch.ones_like(labels)
        sample_weights[labels == -1] = 0

        # y_labels would be same as self.self.encodings i.e input tokens
        y_labels = torch.clone(self.encodings)

        return encodings_masked, self.encodings, sample_weights

    def singular_mask(self):
        mask_encoding, new_labels, new_weights = [torch.empty(0,dtype=torch.int16)] * 3
        for sent_id, weight in enumerate(self.weights):
            masked_idx = torch.nonzero(weight)
            if not any(masked_idx):
                continue

            clones = torch.vstack(tuple(self.labels[sent_id].clone() for _ in masked_idx))  # Initial unchanged labels
            new_labels = torch.cat([new_labels, clones])
            temp_weights = torch.zeros_like(clones)

            for idx, masked_id in enumerate(masked_idx):
                clones[idx][masked_id] = self.mask_encoding[sent_id][masked_id]
                temp_weights[idx][masked_id] = 1

            new_weights = torch.cat([new_weights, temp_weights])
            mask_encoding = torch.cat([mask_encoding, clones])

        return mask_encoding, new_labels, new_weights

# train_set, test_set = BaseDataset('laroseda/laroseda_train.json', file_type='json'), BaseDataset(
#     'laroseda/laroseda_test.json',
#     file_type='json')
# DATAFRAME = pd.concat([train_set.df, test_set.df])
# bert_model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
# tokenizer = BertTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
#
# masked_data = BertMasked(DATAFRAME[:100], tokenizer, bert_model, cached=False)


# sg_mask_data = masked_data.singular_mask()

