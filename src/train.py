import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from dataset import TranslationDataset, causal_mask

def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_dataset(config):
    dataset = load_dataset('opus_books', f'{config["src_lang"]}-{config["tgt_lang"]}', split='train')
    src_tokenizer = get_or_build_tokenizer(config, dataset, config['src_lang'])
    tgt_tokenizer = get_or_build_tokenizer(config, dataset, config['tgt_lang'])

    # split the dataset into train and test (90% train, 10% test)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset_raw, test_dataset_raw = random_split(dataset, [train_size, test_size])

    train_dataset = TranslationDataset(train_dataset_raw, src_tokenizer, tgt_tokenizer, config['src_lang'], config['tgt_lang'], config['seq_len'])
    test_dataset = TranslationDataset(test_dataset_raw, src_tokenizer, tgt_tokenizer, config['src_lang'], config['tgt_lang'], config['seq_len'])

    src_max_len = max(len(src_tokenizer.encode(item['translation'][config['src_lang']]).ids) for item in dataset)
    tgt_max_len = max(len(tgt_tokenizer.encode(item['translation'][config['tgt_lang']]).ids) for item in dataset)
    print(f"Max source length: {src_max_len}")
    print(f"Max target length: {tgt_max_len}")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_loader, test_loader, src_tokenizer, tgt_tokenizer


