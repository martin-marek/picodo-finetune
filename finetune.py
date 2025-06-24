# simulate 8 devices
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import jax
import timing
from gemma import sampler as gemma_sampler
from gemma import pretrained as gemma_pretrained


# config
model_variant = 'gemma3-4b-it-int4'
mesh = jax.make_mesh((1, jax.device_count()), ('data', 'model'))

with timing.context('load model'):    
    model, vocab = gemma_pretrained.load_pretrained(model_variant, mesh)
    # jax.debug.visualize_array_sharding(model.embedder.input_embedding.value)

# sampling
with timing.context('sampling'):
    with mesh:
        sampler = gemma_sampler.Sampler(model, vocab, max_seq_len=100)
        input_batch = ["Say 'Hi!':", "Write a Python program that prints 'Hello World!':"]
        output_batch = sampler(input_batch)
        for inpt, output in zip(input_batch, output_batch):
            print(f"Prompt:\n{inpt}\nOutput:\n{output}\n{10*'*'}")
