import torch

# This is the tokenizer; its taken from https://huggingface.co/google/reformer-enwik8
def encode(list_of_strings, pad_token_id=0):
    max_length = max([len(string) for string in list_of_strings])

    attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
    input_ids = torch.full((len(list_of_strings), max_length), pad_token_id, dtype=torch.long)

    for idx, string in enumerate(list_of_strings):
        if not isinstance(string, bytes):
            string = str.encode(string)

        input_ids[idx, :len(string)] = torch.tensor([x + 2 for x in string])
        attention_masks[idx, :len(string)] = 1
        
    return input_ids, attention_masks

