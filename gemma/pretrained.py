import jax
import jax.numpy as jnp
from flax import nnx
import kagglehub
import sentencepiece as spm
import orbax.checkpoint as ocp
from jax.sharding import PartitionSpec as P, NamedSharding
from . import transformer


# helpers
flatten_path = lambda path: jax.tree_util.keystr(path, simple=True, separator='/')
flatten_tree = lambda tree: {flatten_path(path):v for path, v in jax.tree.leaves_with_path(tree)}
print_tree = lambda tree: jax.tree.map_with_path(lambda path, v: print(f'{flatten_path(path)}: {v.shape}'), tree)


def load_pretrained(model_variant, mesh):

    # download weights
    weights_dir = kagglehub.model_download(f'google/gemma-3/flax/{model_variant}')
    ckpt_path = f'{weights_dir}/{model_variant}'
    vocab_path = f'{weights_dir}/tokenizer.model'

    # load tokenizer
    vocab = spm.SentencePieceProcessor()
    vocab.Load(vocab_path)

    # load abstract model
    model_architecture = '_'.join(model_variant.split('-')[:2])
    model_config = getattr(transformer.TransformerConfig, model_architecture)()
    model = nnx.eval_shape(lambda: transformer.Transformer(model_config, rngs=nnx.Rngs(0)))
    model_state = nnx.state(model)

    # load checkpoint metadata
    checkpointer = ocp.StandardCheckpointer()
    checkpoint = checkpointer.metadata(ckpt_path)

    # add checkpoint sharding annotations
    def add_sharding(path, v):
        key = flatten_path(path)
        pspec = None
        if 'input_embedding' in key: pspec = P('data', 'model') # [V, D]
        if '_norm' in key: pspec = P('data') # [D | H]
        if 'attn_vec_einsum' in key: pspec = P('model', None, 'data') # [N, H, D]
        if 'kv_einsum' in key: pspec = P(None, 'model', 'data', None) # [2, n, D, H]
        if 'q_einsum' in key: pspec = P('model', 'data', None) # [N, D, H]
        if 'mlp/linear' in key: pspec = P('model', 'data') # [F, D]
        if 'mlp/gating_einsum' in key: pspec = P(None, 'model', 'data') # [2, F, D]
        # if pspec is None: print(f'WARNING: {key} has no sharding!')
        sharding = None if pspec is None else NamedSharding(mesh, pspec)
        return jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=sharding)
    checkpoint = jax.tree.map_with_path(add_sharding, checkpoint)

    # load checkpoint weights, then flatten the checkpoint keys
    checkpoint = checkpointer.restore(ckpt_path, checkpoint)
    checkpoint = flatten_tree(checkpoint)
    # for k, v in checkpoint.items():
    #     print(k, v.dtype, type(v))

    # transfer weights to model, mapping model layer keys to checkpoint keys
    def get_weights(path, v):
        """maps layer keys from new flax.nnx format to old flax.linen format"""
        val_fn = lambda x: x
        key = flatten_path(path)
        key = f'transformer/{key}'
        key = key.replace('/value', '')
        key = key.replace('layers/', 'layer_')
        key = key.replace('kernel', 'w')
        key = key.replace('mlp/down_proj', 'mlp/linear')
        if '/scale' in key:
            val_fn = lambda x: x+1
        if 'mlp/gate_proj' in key:
            key = key.replace('mlp/gate_proj', 'mlp/gating_einsum')
            val_fn = lambda x: x[0].T
        if 'mlp/up_proj' in key:
            key = key.replace('mlp/up_proj', 'mlp/gating_einsum')
            val_fn = lambda x: x[1].T
        return val_fn(checkpoint[key])
    model_state = jax.tree.map_with_path(get_weights, model_state)
    nnx.update(model, model_state)

    return model, vocab
