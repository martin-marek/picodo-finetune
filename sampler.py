import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial


@flax.struct.dataclass
class SamplingState:
    step: jnp.int32
    tokens: jnp.ndarray # [B, T]
    kv_cache: dict
    done: jnp.ndarray # [B]


@partial(jax.jit, static_argnames=('model_graphdef'))
def _sample_step(state, model_graphdef, model_state, pad_id, eot_id):

    # we pass model_state as non-static arg, to avoid compiling it
    model = nnx.merge(model_graphdef, model_state)

    # sample next token
    last_token = state.tokens[:, state.step, None] # [B, 1]
    logits, kv_cache = model(last_token, state.kv_cache) # [B, 1, V]
    sampled_token = jnp.argmax(logits, axis=-1)[:, 0] # [B]

    # update buffer
    next_token = state.tokens[:, state.step+1]
    update_token = jnp.where((~state.done) & (next_token==pad_id), sampled_token, next_token)
    tokens = state.tokens.at[:, state.step+1].set(update_token)

    # check if sampling is done
    done = state.done | ((next_token==pad_id) & (sampled_token==eot_id))

    return SamplingState(state.step+1, tokens, kv_cache, done)


def sample(model, tokens, pad_id=0, eot_id=106):
    B, T = tokens.shape

    # initialize state
    state = SamplingState(
        step=0,
        tokens=tokens,
        kv_cache=model.init_kv_cache(B, T),
        done=jnp.zeros([B], dtype=jnp.bool_),
    )

    # sample next token inside a while loop
    step_fn = lambda state: _sample_step(state, *nnx.split(model), pad_id, eot_id)
    cond_fn = lambda state: (state.step < T) & jnp.any(~state.done)
    state = jax.lax.while_loop(cond_fn, step_fn, state)

    # extract output sequences
    outputs = []
    for in_seq, out_seq in zip(tokens, state.tokens):
        out_seq = out_seq[jnp.argmax(in_seq==pad_id):]
        if jnp.any(out_seq==pad_id): out_seq = out_seq[:jnp.argmax(out_seq==pad_id)]
        outputs += [out_seq.tolist()]

    return outputs
