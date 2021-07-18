from torch import nn
from torch import optim
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import torch


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.linear(x)


class BERTLM(nn.Module):
    """
    BERT Language Model
    """

    def __init__(self, bert, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """
        super().__init__()
        self.bert = bert
        self.model = MaskedLanguageModel(self.bert.config.hidden_size, vocab_size)

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        last_hidden_state = outputs[0]
        logits = self.model(last_hidden_state)
        return logits


class BertClassifier(nn.Module):
    def __init__(self, bert, h_size, label_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """
        super().__init__()
        self.bert = bert
        self.model = nn.Sequential(
            nn.Linear(768, h_size),
            nn.ReLU(),
            nn.Linear(h_size, label_size),
            nn.Softmax()
        )

    def forward(self, input_ids):
        outputs = self.bert(input_ids).pooler_output
        logits = self.model(outputs)
        return logits


class BertMaskedLanguage(pl.LightningModule):
    """Bert Model for Masked Language.
    """

    def __init__(self, bert_model, class_num, freeze_bert=True):
        super(BertMaskedLanguage, self).__init__()

        self.bert = bert_model
        self.model = BERTLM(self.bert, class_num)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value
                                                    num_training_steps=10,  # len trainloader
                                                    epochs=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        # Start training loop
        x, y, weights = batch
        y = y * weights
        y[y == 0] = -100

        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat.flatten(0, 1), y.flatten())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, weights = batch
        y = y * weights
        y[y == 0] = -100

        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat.flatten(0, 1), y.flatten())
        # acc = FM.accuracy(y_hat.flatten(0,1), y.flatten())
        metrics = {'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test_acc': metrics['val_acc'], 'test_loss': metrics['val_loss']}
        self.log_dict(metrics)


class BertBasicClassifier(pl.LightningModule):

    def __init__(self, bert_model, h_size, class_num, freeze_bert=True):
        super(BertBasicClassifier, self).__init__()

        self.bert = bert_model
        self.model = BertClassifier(self.bert, h_size, class_num)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=100,  # Default value
                                                    num_training_steps=len(self.train_dataloader()),  # len trainloader
                                                    )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        # Start training loop
        x, y = (t.to(batch) for t in batch)

        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = (t.to(batch) for t in batch)

        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torchmetrics.Accuracy()(torch.argmax(y_hat, dim=1), y)
        metrics = {'acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test_acc': metrics['val_acc'], 'test_loss': metrics['val_loss']}
        self.log_dict(metrics)


class Word2VecLinearDebias(pl.LightningModule):

    def __init__(self, h_size, class_num):
        super(Word2VecLinearDebias, self).__init__()
        self.model = nn.Sequential(nn.Linear(300, h_size),
                                   nn.ReLU(),
                                   nn.Linear(h_size, class_num),
                                   nn.Softmax())

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=100,  # Default value
                                                    num_training_steps=len(self.train_dataloader()),  # len trainloader
                                                    )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        # Start training loop
        x, y = (t for t in batch)

        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = (t for t in batch)

        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torchmetrics.Accuracy()(torch.argmax(y_hat, dim=1), y)
        metrics = {'acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test_acc': metrics['val_acc'], 'test_loss': metrics['val_loss']}
        self.log_dict(metrics)
