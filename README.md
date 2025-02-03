# Aluisio's Translation Pre-trained Transformer (alu-tpt)

![Screencastfrom2025-01-2717-09-55-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/51fda79f-d67e-40de-a6eb-33300a64eca9)

`alu-tpt` is a personal project aimed at deepening my understanding of the Transformer architecture. The implementation is based on the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper and [Umar Jamil's tutorial](https://www.youtube.com/watch?v=ISNdQcPhsts). Every core building block of the Transformer model was implemented from scratch, making the architecture far less abstract and more intuitive for me.  

Due to hardware constraints, I couldn't train a large-scale model. However, `alu-tpt` successfully translates some political texts from English to Portuguese. It was trained on a dataset of 25,000 rows from the [Helsinki-NLP News Commentary dataset](https://huggingface.co/datasets/Helsinki-NLP/news_commentary).

---

Training loss curve:
![image](https://github.com/user-attachments/assets/3dd10927-6609-4c4e-966b-0bd31ee50f74)

Hiperparameters:
```json
{
   "src_lang":"en",
   "tgt_lang":"pt",
   "learning_rate":"10**-4",
   "seq_len":241,
   "d_model":512,
   "num_heads":8,
   "num_layers":6,
   "d_ff":2048,
   "dropout":0.1,
   "batch_size":16,
   "num_epochs":40,
   "max_grad_norm":1.0,
   "warmup_steps":4000
}
```

Personal Notes on Transformers:

- [Transformer Building Blocks](/docs/transformer-blocks.md)

Improvements:

- Learning rate warmup
- Gradient clipping
- Gradient accumulation
- Sub word tokenization
