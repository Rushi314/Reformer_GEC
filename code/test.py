import torch
from pytorch_lightning.utilities import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from code.models.deletion_reformer import DeletionReformer
from code.models.insertion_reformer import InsertionReformer


def main(args):
    file_path = ""  # Add a path to file from which data is to be read.
    pred_file_path = ""  # Add a path to file where the prediction should be written.
    try: file = open(file_path)
    except: print("The input_file path is incorrect!!")
    file_contents = file.read()
    test_data = file_contents.splitlines()

    del_reformer = DeletionReformer.load_from_checkpoint(
        checkpoint_path="/content/gdrive/Shareddrives/CS685_Reformer_GEC/Experiments/DeletionReformer/version_4/checkpoints/epoch=8-val_loss=0.053.ckpt")
    ins_reformer = InsertionReformer.load_from_checkpoint(
        checkpoint_path="/content/gdrive/Shareddrives/CS685_Reformer_GEC/Experiments/InsertionReformer/version_11/checkpoints/epoch=19-val_loss=4.597.ckpt")
    data_loader = DataLoader(test_data, batch_size=len(test_data))  # Can change the batch size later
    try: pred_file = open(pred_file_path, "w")
    except: print("The prediction file path is incorrect!!")

    with torch.no_grad():
        for batch in data_loader:
            del_out = del_reformer(DeletionReformer.encode(batch))
            print(del_out)
            result, temp = [], []
            for pred in del_out:
                insert_out = [pred]
                while insert_out != temp:
                    temp = insert_out
                    insert_out = ins_reformer(InsertionReformer.encode(insert_out))
                print(insert_out[0])
                pred_file.write(insert_out[0] + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GEC_test')

    print(parser.parse_args())
    main(parser.parse_args())