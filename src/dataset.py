import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, dataset, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, seq_len):
        super().__init__()
        
        self.dataset = dataset  
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.Tensor([src_tokenizer.token_to_id("[SOS]")]).long()
        self.eos_token = torch.Tensor([src_tokenizer.token_to_id("[EOS]")]).long()
        self.pad_token = torch.Tensor([src_tokenizer.token_to_id("[PAD]")]).long()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        src_text = self.dataset[idx]['translation'][self.src_lang]
        tgt_text = self.dataset[idx]['translation'][self.tgt_lang]
        
        encoded_src_tokens = self.src_tokenizer.encode(src_text).ids
        encoded_tgt_tokens = self.tgt_tokenizer.encode(tgt_text).ids
         
        src_padding = self.seq_len - len(encoded_src_tokens) - 2
        tgt_padding = self.seq_len - len(encoded_tgt_tokens) - 1

        if src_padding < 0 or tgt_padding < 0:
            raise ValueError("Sentence is too long for the model")

        encoder_input = torch.cat([self.sos_token, torch.Tensor(encoded_src_tokens).long(), self.eos_token, self.pad_token.repeat(src_padding)]).long()
        decoder_input = torch.cat([self.sos_token, torch.Tensor(encoded_tgt_tokens).long(), self.pad_token.repeat(tgt_padding)]).long()
        label = torch.cat([torch.Tensor(encoded_tgt_tokens).long(), self.eos_token, self.pad_token.repeat(tgt_padding)]).long()

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input': encoder_input, # (seq_len)
            'decoder_input': decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len)
            'label': label, # (seq_len)
            'src_text': src_text,
            'tgt_text': tgt_text
        }

def causal_mask(seq_len):
    """
    This function returns a 2D tensor of shape (seq_len, seq_len) where the element at row i and column j is 0 if i > j and 1 otherwise.

    The function is used to prevent the decoder from attending to future tokens during training.
    A word should only attend to words that came before it.
    """
    mask = torch.ones(1, seq_len, seq_len)
    mask = torch.triu(mask, diagonal=1).type(torch.int)
    return mask == 0