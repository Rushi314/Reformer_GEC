import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from itertools import zip_longest
import math

from code.models.reformer import pretrained_reformer

class InsertionReformer(pl.LightningModule):
    # Constructor; creates Reformer and linear layer
    def __init__(self):
        super().__init__()

        self.num_classes = 257 # 256 characters + 1 for no-insert

        self.reformer = pretrained_reformer()

        # Add start and end tokens
        embeddings = self.reformer.get_input_embeddings()
        self.reformer.resize_token_embeddings(embeddings.num_embeddings + 2)

        self.bias_proj = nn.Linear(4096, self.num_classes)
        self.logit_proj = nn.Linear(4096, self.num_classes)


    # A method I made to avoid writing the same code twice
    def shared_forward(self, x):
        input_ids, attention_masks = x
        # input_ids: sentences, except we mapped each character to an integer index
        # attention_masks: tells Reformer where the padding is (because variable length sentences)

        batch_size = attention_masks.shape[0]

        reformer_output = self.reformer(input_ids, attention_masks)['last_hidden_state']
        slots = torch.cat((reformer_output[:, :-1, :], reformer_output[:, 1:, :]), dim=2) # (batch_size, seq_len + 1, hidden_size)

        g = torch.max(slots, dim=1, keepdim=True).values
        b = self.bias_proj(g) # (batch_size, 1, num_classes)

        logits = self.logit_proj(slots) + b # (batch_size, seq_len + 1, num_classes)

        # Slots for padded outputs shouldn't contribute to softmax, so we set them to negative infinity
        mask = torch.where(torch.logical_not(attention_masks)[:, 1:], float('-inf'), float(0)).unsqueeze(2)
        logits += mask

        # Return log probabilities
        return F.log_softmax(logits.view(batch_size, -1), dim=1).view(batch_size, -1, self.num_classes)

    # Tells PyTorch Lightning how to do inference
    def forward(self, x):
        insertion_log_probs = self.shared_forward(x)
        encoded_insertions = torch.argmax(insertion_log_probs, dim=2)

        input_ids, attention_masks = x

        strings = []
        string_lengths = torch.sum(attention_masks, dim=1) - 2
        for encoded_string, string_insertions, length in zip(input_ids, encoded_insertions, string_lengths):
            string = []
            for encoded_char, insertion in zip_longest(encoded_string[1:length + 1], string_insertions[:length + 1]):
                if insertion != 256:
                    # Map class directly back to character via chr()
                    string.append(chr(insertion))
                if encoded_char:
                    # For some reason, the reformer was trained to offset id of each character by 2
                    string.append(chr(encoded_char - 2))

            strings.append("".join(string))
        return strings

    # Tells PyTorch Lightning how to do a training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        log_probs = self.shared_forward(x)

        indices, positions, chars, weights = y
        loss = torch.sum(log_probs[indices, positions, chars] * weights) / torch.sum(weights)

        self.log('train_loss', loss)
        
        return loss

    # Tells PyTorch Lightning how to do a training step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        log_probs = self.shared_forward(x)

        indices, positions, chars, weights = y
        loss = torch.sum(log_probs[indices, positions, chars] * weights) / torch.sum(weights)
        
        self.log('val_loss', loss)

    # Tells PyTorch Lightning which optimizer to use
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    # Collate function for PyTorch DataLoader
    def collate_fn(batch):
        sentences = [datum[0] for datum in batch]
        labels = [datum[1] for datum in batch]

        input_ids, attention_masks = InsertionReformer.encode(sentences)

        indices = []    # Item index from batch
        positions = []  # Position in sentence
        chars = []      # Characters to insert (or no-insert)
        weights = []    # Weight toward total loss (lower for multiple insertion options)
        for index, label in enumerate(labels):
            for position, pos_chars in enumerate(label):
                pos_chars = (None,) if pos_chars is None else pos_chars
                for char in pos_chars:
                    if char is not None:
                        encoded_char = str.encode(char)
                        assert len(encoded_char) == 1 # Makes sure the character is supported

                    indices.append(index)
                    positions.append(position)
                    chars.append(256 if char is None else int.from_bytes(str.encode(char), byteorder='big'))
                    weights.append(1 / len(pos_chars))

        x = (input_ids, attention_masks)
        y = (torch.tensor(indices), torch.tensor(positions), torch.tensor(chars), torch.tensor(weights))

        return (x, y)

    # This is the tokenizer taken from https://huggingface.co/google/reformer-enwik8
    # Modified to add start and end tokens
    def encode(list_of_strings, pad_token_id=0):
        max_length = max([len(string) for string in list_of_strings]) + 2

        # ValueError: If training, sequence length 200 has to be a multiple of least common multiple chunk_length 256. Please consider padding the input to a length of 256.
        max_length = math.ceil(max_length / 256) * 256

        attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
        input_ids = torch.full((len(list_of_strings), max_length), pad_token_id, dtype=torch.long)

        for idx, string in enumerate(list_of_strings):
            if not isinstance(string, bytes):
                string = str.encode(string)

            input_ids[idx, 0] = 258 # <s>
            input_ids[idx, 1:len(string)+1] = torch.tensor([x + 2 for x in string])
            input_ids[idx, len(string)+1] = 259  # </s>
            attention_masks[idx, :len(string)+2] = 1
            
        return input_ids, attention_masks

    def decode(input_ids, attention_masks):
        strings = []
        string_lengths = torch.sum(attention_masks, dim=1) - 2
        for encoded_string, length in zip(input_ids, string_lengths):
            strings.append("".join([chr(x - 2) if x > 1 and x < 258 else "" for x in encoded_string]))
        return strings