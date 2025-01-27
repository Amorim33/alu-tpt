import torch

from config import get_config, get_weights_file_path
from train import get_dataset, get_model, run_test

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = get_config()
    train_loader, test_loader, src_tokenizer, tgt_tokenizer = get_dataset(config)
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)

    model_path = get_weights_file_path(config, config['preload'])
    print(f"Loading model from {model_path}")
    state = torch.load(model_path)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    run_test(model, test_loader, src_tokenizer, tgt_tokenizer, config['seq_len'], device, print)
