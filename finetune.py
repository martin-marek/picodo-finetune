# simulate 8 devices
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

import jax
import timing
import kagglehub
import sentencepiece as spm
from gemma import sampler as gemma_sampler
from gemma import pretrained as gemma_pretrained


# config
model_variant = 'gemma3-1b-it-int4'
mesh = jax.make_mesh((1, jax.device_count()), ('data', 'model'))

with timing.context('download weights'):
    weights_dir = kagglehub.model_download(f'google/gemma-3/flax/{model_variant}')
    ckpt_path = f'{weights_dir}/{model_variant}'
    vocab_path = f'{weights_dir}/tokenizer.model'

with timing.context('load model'):
    model_architecture = '_'.join(model_variant.split('-')[:2])
    model = gemma_pretrained.load_pretrained(model_architecture, ckpt_path, mesh)

# load tokenizer
with timing.context('load tokenizer'):
    vocab = spm.SentencePieceProcessor()
    vocab.Load(vocab_path)

# sampling
with timing.context('sampling'):
    sampler = gemma_sampler.Sampler(transformer=model, vocab=vocab)
    input_batch = ["Say 'Hi!':", "Write a Python program that prints 'Hello World!':"]
    out_text = sampler(input_strings=input_batch, max_seq_len=20)
    for input_string, out_string in zip(input_batch, out_text):
        print(f"Prompt:\n{input_string}\nOutput:\n{out_string}")
        print(10*'#')
