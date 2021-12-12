from transformers import ReformerModel, ReformerConfig
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import math

from code.models.reformer import pretrained_reformer

class DeletionReformer(pl.LightningModule):
    # Constructor; creates Reformer and linear layer
    def __init__(self):
        super().__init__()

        self.reformer = pretrained_reformer()
        self.linear = nn.Linear(2048, 1)

    # A method I made to avoid writing the same code twice
    def shared_forward(self, x):
        input_ids, attention_masks = x
        # input_ids: sentences, except we mapped each character to an integer index
        # attention_masks: tells Reformer where the padding is (because variable length sentences)

        reformer_output = self.reformer(input_ids, attention_masks)['last_hidden_state']
        output = self.linear(reformer_output)

        batch_size = len(input_ids)
        return output.view(batch_size, -1)

    # Tells PyTorch Lightning how to do inference
    def forward(self, x):
        output = self.shared_forward(x)

        # 1 if prob < 0.5, 0 if prob >= 0.5
        deletion_mask = output < 0

        # Return decoded input with characters removed
        return DeletionReformer.decode(x[0] * deletion_mask, x[1])

    # Tells PyTorch Lightning how to do a training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.shared_forward(x)

        labels, label_masks = y
        loss = F.binary_cross_entropy_with_logits(preds, labels, label_masks)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.shared_forward(x)

        labels, label_masks = y
        loss = F.binary_cross_entropy_with_logits(preds, labels, label_masks)
        
        self.log("val_loss", loss)

    # Tells PyTorch Lightning which optimizer to use
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    # Collate function for PyTorch DataLoader
    def collate_fn(batch):
        sentences = [datum[0] for datum in batch]
        labels = [datum[1] for datum in batch]

        input_ids, attention_masks = DeletionReformer.encode(sentences)

        # Add a dummy label so that input padding matches label padding
        labels.append(torch.ones(input_ids.shape[1]))

        padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)[:-1]
        label_masks = nn.utils.rnn.pad_sequence([torch.ones_like(label) for label in labels], batch_first=True)[:-1]

        x = (input_ids, attention_masks)
        y = (padded_labels, label_masks)

        return (x, y)

    # This is the tokenizer taken from https://huggingface.co/google/reformer-enwik8
    def encode(list_of_strings, pad_token_id=0):
        max_length = max([len(string) for string in list_of_strings])

        # ValueError: If training, sequence length 200 has to be a multiple of least common multiple chunk_length 256. Please consider padding the input to a length of 256.
        max_length = math.ceil(max_length / 256) * 256

        attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
        input_ids = torch.full((len(list_of_strings), max_length), pad_token_id, dtype=torch.long)

        for idx, string in enumerate(list_of_strings):
            if not isinstance(string, bytes):
                string = str.encode(string)

            input_ids[idx, :len(string)] = torch.tensor([x + 2 for x in string])
            attention_masks[idx, :len(string)] = 1

        return input_ids, attention_masks

    def decode(input_ids, attention_masks):
        strings = []
        string_lengths = torch.sum(attention_masks, dim=1)
        for encoded_string, length in zip(input_ids, string_lengths):
            strings.append("".join([chr(x - 2) if x > 1 else "" for x in encoded_string]))
        return strings
