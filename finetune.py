# simulate multiple devices
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

import jax
import timing
import gemma
import data
from sampler import sample

# config
model_variant = 'gemma3-1b-it-int4'
mesh = jax.make_mesh((1, jax.device_count()), ('data', 'model'))

with timing.context('load model'):    
    model, vocab = gemma.load_pretrained(model_variant, mesh)
    # jax.debug.visualize_array_sharding(model.embedder.input_embedding.value)

# sampling
with timing.context('sampling'):
    with mesh:
        promps = ["Say 'Hi!'", "Write a Python program that prints 'Hello World!'"]
        inputs_text = [f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n" for q in promps]
        inputs_tokenized = data.tokenize(inputs_text, vocab, 100)
        outputs_tokens = sample(model, inputs_tokenized)
        outputs_text = vocab.DecodeIds(outputs_tokens)
        for inpt, output in zip(inputs_text, outputs_text):
            print(f'{inpt+output}\n{10*"*"}')
