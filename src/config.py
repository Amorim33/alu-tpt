from pathlib import Path

def get_config():
    return {
        'src_lang': 'en',
        'tgt_lang': 'pt',
        "lr": 10**-3,
        "seq_len": 249,
        'd_model': 512,  
        'batch_size': 16,  
        'num_epochs': 50,  
        'model_folder': 'weights',
        'model_filename': 'alu-tpt',
        "preload": "24",
        'tokenizer_path': 'tokenizer_{0}.json',
        'experiment_name': 'runs/alu-tpt',
        'gradient_accumulation_steps': 8,
        'max_grad_norm': 1.0
    }

def get_weights_file_path(config, epoch: str):
    return str(Path('.') / config['model_folder'] / f"{config['model_filename']}-{epoch}.pt")