import os
from pathlib import Path
from tokenizers import Tokenizer
import warnings
import torch
import time

from config import get_config, get_weights_file_path
from dataset import causal_mask
from train import get_model

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = get_config()
    src_tokenizer = Tokenizer.from_file(str(Path(config['tokenizer_path'].format(config['src_lang']))))
    tgt_tokenizer = Tokenizer.from_file(str(Path(config['tokenizer_path'].format(config['tgt_lang']))))
    
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)
    model_path = get_weights_file_path(config, config['preload'])
    state = torch.load(model_path)
    model.load_state_dict(state['model_state_dict'])

    while True:
        text = input("\nEnter news text (or 'q' to quit): ")
        if text.lower() == 'q':
            break
            
        with torch.no_grad():
            encoded_src_tokens = src_tokenizer.encode(text).ids
            src_padding = config['seq_len'] - len(encoded_src_tokens) - 2
            if src_padding < 0:
                raise ValueError("Sentence is too long for the model")
            
            sos_token = torch.Tensor([src_tokenizer.token_to_id("[SOS]")]).long().to(device)
            eos_token = torch.Tensor([src_tokenizer.token_to_id("[EOS]")]).long().to(device)
            pad_token = torch.Tensor([src_tokenizer.token_to_id("[PAD]")]).long().to(device)

            encoder_input = torch.cat([sos_token, torch.Tensor(encoded_src_tokens).long().to(device), eos_token, pad_token.repeat(src_padding)]).long().to(device)
            encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)

            sos_idx = src_tokenizer.token_to_id("[SOS]")
            eos_idx = src_tokenizer.token_to_id("[EOS]")

            encoder_output = model.encode(encoder_input, encoder_mask)
            
            decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)
            while decoder_input.size(1) != config['seq_len']:
                decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

                proj_output = model.project(decoder_output[:, -1])
                _, next_token = torch.max(proj_output, dim=-1)

                decoder_input = torch.cat(
                    [decoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_token.item()).to(device)], dim=1
                )

                if next_token == eos_idx:
                    break
            
            model_output = decoder_input.squeeze(0)

            model_output_text = tgt_tokenizer.decode(model_output.detach().cpu().numpy())

            print(model_output_text)
            
            time.sleep(2)
            os.system('clear')
