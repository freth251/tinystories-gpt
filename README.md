# tinystories-gpt

A small GPT-style language model trained from scratch on the
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset,
using the GPT-2 tokenizer (tiktoken).

## Files

- `config.py` – model size and training settings
- `model.py` – the transformer model (attention, MLP blocks, generation)
- `prepare.py` – downloads TinyStories and tokenizes it into `train.bin` / `val.bin`
- `train.py` – trains the model and saves checkpoints to `ckpt.pt`
- `sample.py` – loads `ckpt.pt` and generates text

## Usage

```bash
pip install -r requirements.txt
python prepare.py
python train.py
python sample.py 300
```

`sample.py` takes the number of tokens to generate and an optional starting prompt:
`python sample.py 300 "Once upon a time"`.
