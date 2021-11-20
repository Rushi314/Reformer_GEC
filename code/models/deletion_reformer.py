from transformers import ReformerModel, ReformerConfig
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

from code.models.reformer import encode

class DeletionReformer(pl.LightningModule):
    # Constructor; creates Reformer and linear layer
    def __init__(self):
        super().__init__()

        # We load the pre-trained Reformer's config first because we need to make changes to it
        # https://stackoverflow.com/questions/68742863/error-while-trying-to-fine-tune-the-reformermodelwithlmhead-google-reformer-enw#answer-68885046
        config = ReformerConfig.from_pretrained('google/reformer-enwik8')
        config.is_decoder = False  # change masked self-attention to normal self-attention
        config.num_hashes = 2  # was 4; lowering to 2 reduces memory at expense of accuracy (we can raise it back later)
        config.axial_pos_embds = False  # replace axial position embeddings with new learned embeddings
                                        # this avoids an issue where all inputs needed to be padded to size 65536
        self.reformer = ReformerModel.from_pretrained(
            'google/reformer-enwik8', config=config)

        self.linear = nn.Linear(2048, 1)

    # A method I made to avoid writing the same code twice
    def shared_forward(self, x):
        input_ids, attention_masks = x
        # input_ids: sentences, except we mapped each character to an integer index
        # attention_masks: tells Reformer where the padding is (because variable length sentences)

        reformer_output = self.reformer(input_ids, attention_masks)
        output = self.linear(reformer_output['last_hidden_state'])

        batch_size = len(input_ids)
        return output.view(batch_size, -1)

    # Tells PyTorch Lightning how to do inference
    def forward(self, x):
        output = self.shared_forward(x)
        return torch.sigmoid(output)

    # Tells PyTorch Lightning how to do a training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.shared_forward(x)

        labels, label_masks = y
        loss = F.binary_cross_entropy_with_logits(preds, labels, label_masks)
        return loss

    # Tells PyTorch Lightning which optimizer to use
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    # Collate function for PyTorch DataLoader
    def collate_fn(batch):
        sentences = [datum[0] for datum in batch]
        labels = [datum[1] for datum in batch]

        input_ids, attention_masks = encode(sentences)

        padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
        label_masks = nn.utils.rnn.pad_sequence([torch.ones_like(label) for label in labels], batch_first=True)

        x = (input_ids, attention_masks)
        y = (padded_labels, label_masks)

        return (x, y)