import torch
from transformers import ReformerModel, ReformerConfig, ReformerTokenizer

# Just a few reformer utils

def pretrained_reformer():
    # We load the pre-trained Reformer's config first because we need to make changes to it
    # https://stackoverflow.com/questions/68742863/error-while-trying-to-fine-tune-the-reformermodelwithlmhead-google-reformer-enw#answer-68885046
    config = ReformerConfig.from_pretrained('google/reformer-enwik8')
    config.is_decoder = False  # change masked self-attention to normal self-attention
    config.num_hashes = 2  # was 4; lowering to 2 reduces memory at expense of accuracy (we can raise it back later)
    config.axial_pos_embds = False  # replace axial position embeddings with new learned embeddings
                                    # this avoids an issue where all inputs needed to be padded to size 65536

    reformer = ReformerModel.from_pretrained('google/reformer-enwik8', config=config)

    return reformer
