from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from code.models.deletion_reformer import DeletionReformer
from code.models.insertion_reformer import InsertionReformer
from code.data.load_data import load_del_data, load_ins_data

def main(args):
    if args.model is None:
        raise ValueError("Please specify a model to train using the -m flag")
    elif args.model == 'deletion':
        train_data, val_data = load_del_data()
        model = DeletionReformer()
    elif args.model == 'insertion':
        train_data, val_data = load_ins_data()
        model = InsertionReformer()
    else:
        raise ValueError(f"Model {args.model} is not found")

    batch_size = 32

    train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=type(model).collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, collate_fn=type(model).collate_fn)

    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            filename='{epoch}-{val_loss:.3f}',
            save_last=True
        ),
        EarlyStopping(monitor='val_loss', patience=5)
    ]

    logger = TestTubeLogger("drive/Shareddrives/CS685_Reformer_GEC/Experiments", name=model.__name__)
    # logger = TestTubeLogger("Experiments", name=type(model).__name__)
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        val_check_interval=0.5,
        gpus=1 if torch.cuda.is_available() else 0
    )
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
        ckpt_path=args.checkpoint_path
    )

if __name__ == "__main__":
    parser = ArgumentParser(description='GEC')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='Model to train')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        help='Path of the checkpoint to load')

    print(parser.parse_args())
    main(parser.parse_args())
