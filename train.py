import os
import os.path as osp
import time
import argparse
import yaml
import pickle
import wandb
import glob

import jax
from jax import random
import jax.numpy as jnp
from flax.training import checkpoints
from flax import jax_utils

import optax
from functools import partial

from dataloading import create_wikitext_dataset, create_icl_datasets
from train_utils import init_model_state, \
        get_first_device, ProgressMeter, seed_all, reshape_batch_per_device
from src.models import get_model


def main():
    global model
    rng = random.PRNGKey(config.seed)
    rng, init_rng = random.split(rng)
    seed_all(config.seed)

    files = glob.glob(osp.join(config.output_dir, 'checkpoints', '*'))
    if len(files) > 0:
        print('Found previous checkpoints', files)
        config.ckpt = config.output_dir
    else:
        config.ckpt = None

    if is_master_process:
        root_dir = os.environ['DATA_DIR']
        os.makedirs(osp.join(root_dir, 'wandb'), exist_ok=True)

        wandb.init(project=config.project, entity=config.entity, config=config,
                   dir=root_dir, id=config.run_id, resume='allow')
        wandb.run.name = config.run_id
        wandb.run.save()

    if config.dataset in ["wikitext103"]:
        train_loader, val_loader, test_loader = create_wikitext_dataset(config)
    elif config.dataset in ["icl_synthetics"]:
        train_loader, val_loader, test_loader = create_icl_datasets(config)
    else:
        raise NotImplementedError("Dataset not implemented")
    log_metrics = ['loss', 'accuracy']

    batch = next(iter(train_loader))
    inputs = jnp.array(batch[0].numpy())
    targets = jnp.array(batch[1].numpy())

    num_devices = jax.local_device_count()
    batch_size_per_device = inputs.shape[0]
    if config.layer == "S5_operator": # B * layers * order * 1 (time) * ssm_size
        zero_hiddens = jax.numpy.zeros((batch_size_per_device, config.n_layer, config.layer_kwargs["order"], 1, config.layer_kwargs["ssm_size"]))
        zero_hiddens_init_model_state = jax.numpy.zeros((batch_size_per_device // num_devices, config.n_layer, config.layer_kwargs["order"], 1, config.layer_kwargs["ssm_size"]))
    elif config.layer == "hyena":
        zero_hiddens = jax.numpy.zeros((batch_size_per_device, config.n_layer, config.layer_kwargs["order"], 1, config.d_model, 1, 1))
        zero_hiddens_init_model_state = None
    elif config.layer == "Mamba_operator": # B * layers * 1 (time) * d_inner * d_state
        config.layer_kwargs["d_inner"] = config.d_model * config.layer_kwargs["expand"]

        zero_hiddens = jax.numpy.zeros((batch_size_per_device, config.n_layer, 1, config.layer_kwargs["d_inner"], config.layer_kwargs["d_state"]))
        zero_hiddens_init_model_state = jax.numpy.zeros((batch_size_per_device // num_devices, config.n_layer, 1, config.layer_kwargs["d_inner"], config.layer_kwargs["d_state"]))
    elif config.layer == "LSTM_operator": # B * layers * 1 (time) * d_model
        zero_hiddens = jax.numpy.zeros((batch_size_per_device, config.n_layer, config.d_model, 2))
        zero_hiddens_init_model_state = jax.numpy.zeros((batch_size_per_device // num_devices, config.n_layer, config.d_model, 2))
    elif config.layer == "GRU_operator": # B * layers * 1 (time) * d_model
        zero_hiddens = jax.numpy.zeros((batch_size_per_device, config.n_layer, config.d_model))
        zero_hiddens_init_model_state = jax.numpy.zeros((batch_size_per_device // num_devices, config.n_layer, config.d_model))
    else:
        raise NotImplementedError(f"Hidden state initialization for {config.layer} not implemented")

    # Reshape to (num_devices, device_batch_size, seq_len, dim)
    num_devices = jax.local_device_count()
    inputs = reshape_batch_per_device(inputs, num_devices)
    targets = reshape_batch_per_device(targets, num_devices)
    batch = (inputs, targets)  # Just want to use 1 device batch for init

    batch = get_first_device(batch)
    model = get_model(config)
    state, _, schedule_fn = init_model_state(init_rng, model, batch[0], zero_hiddens_init_model_state, config)
    if config.ckpt is not None:
        state = checkpoints.restore_checkpoint(osp.join(config.ckpt, 'checkpoints'), state)
        print('Restored from checkpoint')

    iteration = int(state.step)
    state = jax_utils.replicate(state)

    ckpt_dir = osp.join(config.output_dir, 'checkpoints')

    rngs = random.split(rng, jax.local_device_count())

    p_train_step = jax.pmap(partial(train_step, config=config, vocab_size=config.vocab_size), axis_name='batch')
    if config.dataset in ["wikitext103"]:
        p_eval_step = jax.pmap(partial(eval_step, config=config, vocab_size=config.vocab_size), axis_name='batch')
    elif config.dataset in ["icl_synthetics"]:
        pass
        # p_eval_step = jax.pmap(partial(eval_step_synthetic, config=config, vocab_size=config.vocab_size), axis_name='batch')
    else:
        raise NotImplementedError("Dataset not implemented")

    # with jax.disable_jit():
    while iteration <= config.total_steps:
        iteration, state, rngs = train(config, iteration, log_metrics, state, zero_hiddens, train_loader,
                                    schedule_fn, rngs, ckpt_dir, p_train_step=p_train_step)

        rngs, avg_loss, avg_perplexity, avg_accuracy = validate(config, iteration, state, zero_hiddens, val_loader, rngs, val=1)
        print("For validation set", avg_loss, avg_perplexity, avg_accuracy, p_eval_step=p_eval_step)
        rngs, avg_loss, avg_perplexity, avg_accuracy = validate(config, iteration, state, zero_hiddens, test_loader, rngs, val=2)
        print("For test set", avg_loss, avg_perplexity, avg_accuracy, p_eval_step=p_eval_step)


def train_step(config, batch, state, hiddens, rng, vocab_size):
    new_rng, *rngs = random.split(rng, len(config.rng_keys) + 1)
    rngs = {k: r for k, r in zip(config.rng_keys, rngs)}

    inputs = batch[0]
    targets = batch[1]

    def loss_fn(params, hiddens):
        variables = {'params': params, **state.model_state}
        out = state.apply_fn(
            variables,
            inputs,
            hiddens,
            training=True,
            rngs=rngs
        )
        out_tuple, hiddens = out
        logits = out_tuple.logits
        labels = jax.nn.one_hot(targets, num_classes=vocab_size)

        loss = optax.softmax_cross_entropy(logits, labels)
        loss = loss.mean()
        preds = jnp.argmax(logits, axis=-1)
        accuracy = (preds == targets).mean()
        perplexity = jnp.exp(loss)
        out_dict = {'loss': loss,
                    'accuracy': accuracy,
                    'perplexity': perplexity
                    }

        return loss, (out_dict, hiddens)

    (loss, (out_dict, hiddens)), grads = jax.value_and_grad(loss_fn,
                                            has_aux=True)(state.params, hiddens)
    grads = jax.lax.pmean(grads, axis_name='batch')

    def norm(x):
        return jnp.sqrt(jnp.sum(x**2))
    
    def grad_over_weight_max(x, y, epsilon=1e-9):
        grad_over_weight = jnp.abs(x) / (jnp.abs(y) + epsilon)
        return grad_over_weight.max()
    
    def grad_over_weight_min(x, y, epsilon=1e-9):
        grad_over_weight = jnp.abs(x) / (jnp.abs(y) + epsilon)
        return grad_over_weight.min()
    
    g_norms = jax.tree_map(norm, grads)
    gow_maxs = jax.tree_map(grad_over_weight_max, grads, state.params)
    gow_mins = jax.tree_map(grad_over_weight_min, grads, state.params)

    new_state = state.apply_gradients(
        grads=grads,
    )

    return new_state, hiddens, out_dict, new_rng, g_norms, gow_maxs, gow_mins


def flatten_tree_with_names(tree):
    # Flatten the tree with paths
    flattened_with_paths, _ = jax.tree_util.tree_flatten_with_path(tree)
    
    # Create a dictionary with the concatenated path as the key
    flat_dict = {'/'.join(map(str, path)): value for path, value in flattened_with_paths}
    
    return flat_dict


def train(config, iteration, log_metrics, state, hiddens, train_loader, schedule_fn, rngs, ckpt_dir, p_train_step=None):
    progress = ProgressMeter(config.total_steps,
                             ['time', 'data'] + log_metrics)
    is_master_process = jax.process_index() == 0

    num_devices = jax.local_device_count()

    end = time.time()
    for batch in train_loader:
        inputs = jnp.array(batch[0].numpy())
        targets = jnp.array(batch[1].numpy())

        # Reshape to (num_devices, device_batch_size, seq_len, dim)
        inputs = reshape_batch_per_device(inputs, num_devices)
        targets = reshape_batch_per_device(targets, num_devices)
        batch = (inputs, targets)
        hiddens = reshape_batch_per_device(hiddens, num_devices)

        batch_size = batch[0].shape[1]
        progress.update(data=time.time() - end)

        if config.layer == "S5_operator":
            if len(hiddens.shape) > 6:
                hiddens = jax.numpy.squeeze(hiddens, axis=-6)
        elif config.layer == "hyena":
            if len(hiddens.shape) > 8:
                hiddens = jax.numpy.squeeze(hiddens, axis=-8)
        elif config.layer == "Mamba_operator":
            if len(hiddens.shape) > 6:
                hiddens = jax.numpy.squeeze(hiddens, axis=-6)
        elif config.layer == "LSTM_operator":
            if len(hiddens.shape) > 5:
                hiddens = jax.numpy.squeeze(hiddens, axis=-5)
        elif config.layer == "GRU_operator":
            if len(hiddens.shape) > 4:
                hiddens = jax.numpy.squeeze(hiddens, axis=-4)
        else:
            raise NotImplementedError(f"Hidden state initialization for {config.layer} not implemented")

        if iteration < 3:
            print(f"hiddens shape {hiddens.shape}")

        state, hiddens, return_dict, rngs, g_norms, gow_maxs, gow_mins = p_train_step(batch=batch, state=state, hiddens=hiddens, rng=rngs)

        metrics = {k: return_dict[k].mean() for k in log_metrics}
        metrics = {k: v.astype(jnp.float32) for k, v in metrics.items()}
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})

        if is_master_process and iteration % config.log_interval == 0:
            wandb.log({'train/lr': schedule_fn(iteration)}, step=iteration)
            wandb.log({**{f'train/{metric}': val for metric, val in metrics.items()}}, step=iteration)
            wandb.log({**{f'grad_norm/{layer}': jnp.linalg.norm(norm) for layer, norm in flatten_tree_with_names(g_norms).items()}}, step=iteration)
            wandb.log({**{f'gow_max/{layer}': jnp.linalg.norm(norm) for layer, norm in flatten_tree_with_names(gow_maxs).items()}}, step=iteration)
            wandb.log({**{f'gow_min/{layer}': jnp.linalg.norm(norm) for layer, norm in flatten_tree_with_names(gow_mins).items()}}, step=iteration)

        if iteration % config.log_interval == 0:
            progress.display(iteration)

        if iteration <= 12:
            progress.display(iteration)

        if iteration % config.save_interval == 0:
            if is_master_process:
                state_ = jax_utils.unreplicate(state)
                save_path = checkpoints.save_checkpoint(ckpt_dir, state_, state_.step, keep=1)
                print('Saved checkpoint to', save_path)
                del state_  # Needed to prevent a memory leak bug

        progress.update(time=time.time() - end)
        end = time.time()

        iteration += 1

    if is_master_process:
        state_ = jax_utils.unreplicate(state)
        save_path = checkpoints.save_checkpoint(ckpt_dir, state_, state_.step, keep=1)
        print('Saved checkpoint to', save_path)
        del state_  # Needed to prevent a memory leak bug

    return iteration, state, rngs


def eval_step(config, batch, state, hiddens, rng, vocab_size):
    new_rng, *rngs = random.split(rng, len(config.rng_keys) + 1)
    rngs = {k: r for k, r in zip(config.rng_keys, rngs)}

    inputs = batch[0]
    targets = batch[1]

    variables = {'params': state.params, **state.model_state}
    out = state.apply_fn(
        variables,
        inputs,
        hiddens,
        training=False,
        rngs=rngs
    )
    out_tuple, hiddens = out
    logits = out_tuple.logits
    labels = jax.nn.one_hot(targets, num_classes=vocab_size)

    loss = optax.softmax_cross_entropy(logits, labels)
    preds = jnp.argmax(logits, axis=-1)
    accuracy = (preds == targets)
    return loss, accuracy, hiddens, new_rng


def eval_step_synthetic(config, batch, state, vocab_size):
    """Different eval loss functions for
       synthetic associative_recall task"""
    inputs = batch[0]
    targets = batch[1]

    variables = {'params': state.params, **state.model_state}
    out = state.apply_fn(
        variables,
        inputs,
        training=False,
    )
    out_tuple, _ = out
    logits = out_tuple.logits[:, -1]
    labels = jax.nn.one_hot(targets[:, -1], num_classes=vocab_size)

    loss = optax.softmax_cross_entropy(logits, labels)
    preds = jnp.argmax(logits, axis=-1)
    accuracy = (preds == targets[:, -1])

    return loss, accuracy


def validate(config, iteration, state, hiddens, test_loader, rngs, val=0, seq_len=None, p_eval_step=None):
    losses = jnp.array([])
    accs = jnp.array([])
    is_master_process = jax.process_index() == 0

    # Todo: may need to change for multinode
    num_devices = jax.local_device_count()

    for batch in test_loader:
        inputs = jnp.array(batch[0].numpy())
        targets = jnp.array(batch[1].numpy())
        if inputs.shape[0] < config.data_kwargs["batch_size_eval"]:
            continue # TODO, for correctness purpose need to modify the evaluation. 

        # Reshape to (num_devices, device_batch_size, seq_len, dim)
        inputs = reshape_batch_per_device(inputs, num_devices)
        targets = reshape_batch_per_device(targets, num_devices)
        batch = (inputs, targets)
        hiddens = reshape_batch_per_device(hiddens, num_devices)

        if config.layer == "S5_operator":
            if len(hiddens.shape) > 6:
                hiddens = jax.numpy.squeeze(hiddens, axis=-6)
        elif config.layer == "hyena":
            if len(hiddens.shape) > 8:
                hiddens = jax.numpy.squeeze(hiddens, axis=-8)
        elif config.layer == "Mamba_operator":
            if len(hiddens.shape) > 6:
                hiddens = jax.numpy.squeeze(hiddens, axis=-6)
        elif config.layer == "LSTM_operator":
            if len(hiddens.shape) > 5:
                hiddens = jax.numpy.squeeze(hiddens, axis=-5)
        elif config.layer == "GRU_operator":
            if len(hiddens.shape) > 4:
                hiddens = jax.numpy.squeeze(hiddens, axis=-4)
        else:
            raise NotImplementedError(f"Hidden state shape modification for {config.layer} not implemented")

        return_loss, return_acc, hiddens, rngs = p_eval_step(batch=batch, state=state, hiddens=hiddens, rng=rngs)
        losses = jnp.append(losses, return_loss)
        accs = jnp.append(accs, return_acc)

    avg_loss = jnp.mean(losses)
    avg_perplexity = jnp.exp(avg_loss)
    avg_accuracy = jnp.mean(accs)
    if is_master_process:
        if val == 0:
            prefix = "train"
        elif val == 1:
            prefix = "val"
        elif val == 2:
            prefix = "test"
        else:
            raise ValueError('val must be 0, 1, or 2')

        if seq_len is not None:
            prefix = prefix + f"/seq_len_{seq_len}"

        print(prefix+'/loss:', avg_loss)
        print(prefix + '/perplexity:', avg_perplexity)
        print(prefix + '/accuracy:', avg_accuracy)

        wandb.log({prefix+'/loss': avg_loss}, step=iteration)
        wandb.log({prefix+'/perplexity': avg_perplexity}, step=iteration)
        wandb.log({prefix + '/accuracy': avg_accuracy}, step=iteration)

    return rngs, avg_loss, avg_perplexity, avg_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-l', '--train_length', type=int, required=False, default=2048)
    args = parser.parse_args()

    # add the train_length to the end of the output_dir
    args.output_dir = args.output_dir + f"_T{args.train_length}"
    print("The training length is", args.train_length)

    args.run_id = args.output_dir

    print(f'JAX process: {jax.process_index()} / {jax.process_count()}')
    print(f'JAX total devices: {jax.device_count()}')
    print(f'JAX local devices: {jax.local_device_count()}')

    if not osp.isabs(args.output_dir):
        if 'DATA_DIR' not in os.environ:
            os.environ['DATA_DIR'] = 'logs'
            print('DATA_DIR environment variable not set, default to logs/')
        root_folder = os.environ['DATA_DIR']
        args.output_dir = osp.join(root_folder, args.output_dir)

    config = yaml.safe_load(open(args.config, 'r'))
    if os.environ.get('DEBUG') == '1':
        config['save_interval'] = 2
        config['log_interval'] = 1
        args.output_dir = osp.join(osp.dirname(args.output_dir), f'DEBUG_{osp.basename(args.output_dir)}')
        args.run_id = f'DEBUG_{args.run_id}'

    print(f"Logging to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    args_d = vars(args)
    args_d.update(config)
    pickle.dump(args, open(osp.join(args.output_dir, 'args'), 'wb'))
    config = args

    is_master_process = jax.process_index() == 0

    # adjust the batch size if the training sequence length is changed 
    config.data_kwargs["batch_size"] = config.l_max * config.data_kwargs["batch_size"] // config.train_length
    config.data_kwargs["batch_size_eval"] = config.l_max * config.data_kwargs["batch_size_eval"] // config.train_length
    assert config.data_kwargs["batch_size_eval"] > 0
    assert config.data_kwargs["batch_size"] % jax.device_count() == 0, "Batch size must be divisible by the number of devices"
    config.l_max = config.train_length

    main()
