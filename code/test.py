import torch
from pytorch_lightning.utilities import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from code.models.deletion_reformer import DeletionReformer
from code.models.insertion_reformer import InsertionReformer


def main(args):
    test_data = ["Childen play footbl", "Tody ia a good day!", "How u doing?"]

    del_reformer = DeletionReformer.load_from_checkpoint(checkpoint_path="deletion_logs/lightning_logs/version_0/checkpoints/epoch=1-step=11.ckpt")
    ins_reformer = InsertionReformer.load_from_checkpoint(checkpoint_path="insertion_logs/lightning_logs/version_1/checkpoints/epoch=1-step=7.ckpt")
    data_loader = DataLoader(test_data, batch_size=len(test_data)) # Can change the batch size later

    with torch.no_grad():
        for batch in data_loader:
            del_out = del_reformer(DeletionReformer.encode(batch))
            print(del_out)
            result, temp = [], ['']
            for pred in del_out:
                insert_out = pred
                while insert_out != temp:
                    temp = insert_out
                    insert_out = ins_reformer(InsertionReformer.encode(insert_out))
                result.append(insert_out)
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GEC_test')

    print(parser.parse_args())
    main(parser.parse_args())