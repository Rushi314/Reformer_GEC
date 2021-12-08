import torch
from pytorch_lightning.utilities import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from code.models.deletion_reformer import DeletionReformer
from code.models.insertion_reformer import InsertionReformer


def main(args):
    data_ins = [
        ('hello', [None,None,None,None,None,None]),
        ('hi', [None,None,None]),
        ('hello', [None, None, None, None, None, None]),
        ('hi', [None, None, None]),
        ('hello', [None, None, None, None, None, None]),
        ('hi', [None, None, None]),
        ('hello', [None, None, None, None, None, None]),
        ('hi', [None, None, None]),
        ('hello', [None, None, None, None, None, None]),
        ('hi', [None, None, None]),
        ('hello', [None, None, None, None, None, None]),
        ('hi', [None, None, None])
    ]
    data = [
        ('hello', torch.zeros(5)),
        ('hi', torch.zeros(2)),
        ('hello', torch.zeros(5)),
        ('hi', torch.zeros(2)),
        ('hello', torch.zeros(5)),
        ('hi', torch.zeros(2)),
        ('hello', torch.zeros(5)),
        ('hi', torch.zeros(2)),
        ('hello', torch.zeros(5)),
        ('hi', torch.zeros(2)),
        ('hello', torch.zeros(5)),
        ('hi', torch.zeros(2))
    ]
    if args.deletion == True:
        del_reformer = DeletionReformer()
        # Fake dataset of the form (ungrammatical sentence, tensor of 0/1s corresponding to keep/delete for each character)
        train_data_loader = DataLoader(data, batch_size=2, collate_fn=DeletionReformer.collate_fn)
        val_data_loader = DataLoader(data, batch_size=2, collate_fn=DeletionReformer.collate_fn)
        # PyTorch Lightning trainer, which automatically handles the training loop and sending data/model to GPU
        trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=args.epochs_del, default_root_dir = "deletion_logs/")
        trainer.fit(del_reformer, train_data_loader, val_data_loader)

    if args.insertion == True:
        ins_reformer = InsertionReformer()
        train_data_loader = DataLoader(data_ins, batch_size=3, collate_fn=InsertionReformer.collate_fn)
        val_data_loader = DataLoader(data_ins, batch_size=3, collate_fn=InsertionReformer.collate_fn)
        trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=args.epochs_ins, default_root_dir = "insertion_logs/")
        trainer.fit(ins_reformer, train_data_loader, val_data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GEC')
    # Here we can add arguments to pass to the trainer
    parser.add_argument("-d", "--deletion", default=False,
                        help="True if you want to train Deletion Transformer False otherwise")
    parser.add_argument("-k", "--epochs_del", default=2,
                        help="Number of epochs to train the deletion reformer")
    parser.add_argument("-l", "--epochs_ins", default=2,
                        help="Number of epochs to train the insertion reformer")
    parser.add_argument("-i", "--insertion", default=True,
                        help="True if you want to train Insertion Transformer False otherwise")
    print(parser.parse_args())
    main(parser.parse_args())
