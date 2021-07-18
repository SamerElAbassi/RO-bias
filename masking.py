# %%
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, AutoModel
import torch.nn.functional as F
from torch import nn
from dataset import BaseDataset, BertDataset, BertTextDataset

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from dataset import BaseDataset, BertMasked
import pickle as pkl
from models import BertMaskedLanguage

pl.seed_everything(42)
device = "cuda"
bert_model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1").to(device)
tokenizer = BertTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
laroseda_data = BaseDataset('laroseda/laroseda_train.json', 'reviews',
                            file_type='json')  # Normalize dataset and change col names
masked_dataset = BertMasked(laroseda_data.df['text'], tokenizer, bert_model)
with open('smaller.corpus', 'rb') as f:
    text = BertTextDataset(pkl.load(f))

sing_masked = masked_dataset.singular_mask()
task = BertMaskedLanguage(bert_model, tokenizer.vocab_size)
