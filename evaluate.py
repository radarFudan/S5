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

from train import train_step, train, eval_step, eval_step_synthetic, validate

import csv


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
    if config.layer == "S5_operator":
        zero_hiddens = jax.numpy.zeros((batch_size_per_device, config.n_layer, config.layer_kwargs["order"], 1, config.layer_kwargs["ssm_size"]))
        zero_hiddens_init_model_state = jax.numpy.zeros((batch_size_per_device // num_devices, config.n_layer, config.layer_kwargs["order"], 1, config.layer_kwargs["ssm_size"]))
    elif config.layer == "hyena":
        zero_hiddens = jax.numpy.zeros((batch_size_per_device, config.n_layer, config.layer_kwargs["order"], 1, config.d_model, 1, 1))
        zero_hiddens_init_model_state = None
    elif config.layer == "Mamba_operator": # B * layers * 1 (time) * d_inner * d_state
        config.layer_kwargs["d_inner"] = config.d_model * config.layer_kwargs["expand"]

        zero_hiddens = jax.numpy.zeros((batch_size_per_device, config.n_layer, 1, config.layer_kwargs["d_inner"], config.layer_kwargs["d_state"]))
        zero_hiddens_init_model_state = jax.numpy.zeros((batch_size_per_device // num_devices, config.n_layer, 1, config.layer_kwargs["d_inner"], config.layer_kwargs["d_state"]))
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
    # while iteration <= config.total_steps:
    #     iteration, state, rngs = train(iteration, log_metrics, state, train_loader,
    #                                    schedule_fn, rngs, ckpt_dir)

    #     validate(iteration, state, val_loader, val=True)

    #     validate(iteration, state, test_loader)

    # Consecutive evaluation
    evaluation(config, rngs, iteration, state, consecutive_loader=True)
    evaluation(config, rngs, iteration, state, consecutive_loader=False)


def evaluation(config, rngs, iteration, state, consecutive_loader=True, evaluate_train=False):
    config.data_kwargs["consecutive_loader"] = consecutive_loader
    print("\nNow we are evaluating with consecutive loader:", consecutive_loader, "\n")

    num_devices = jax.local_device_count()

    # Initialize a list to store validation losses
    evaluation = []

    # seq_len_list = [16 * 2 ** i for i in range(0,12)] # 16 to 32768
    seq_len_list = [16 * 2 ** i for i in range(0,10)] # 16 to 8192
    for seq_len in seq_len_list:
        # Change the sequence length for evaluation
        # TODO, care about the out of memory issue. 
        config.l_max = seq_len
        config.data_kwargs["batch_size"] = max(int(16 * 2048 // seq_len), num_devices)
        config.data_kwargs["batch_size_eval"] = max(int(16 * 2048 // seq_len), num_devices)
        
        train_loader, val_loader, test_loader = create_wikitext_dataset(config)

        batch = next(iter(train_loader))
        inputs = jnp.array(batch[0].numpy())
        targets = jnp.array(batch[1].numpy())
        num_devices = jax.local_device_count()
        batch_size_per_device = inputs.shape[0]
        if config.layer == "S5_operator":
            zero_hiddens = jax.numpy.zeros((batch_size_per_device, config.n_layer, config.layer_kwargs["order"], 1, config.layer_kwargs["ssm_size"]))
            zero_hiddens_init_model_state = jax.numpy.zeros((batch_size_per_device // num_devices, config.n_layer, config.layer_kwargs["order"], 1, config.layer_kwargs["ssm_size"]))
        elif config.layer == "hyena":
            zero_hiddens = jax.numpy.zeros((batch_size_per_device, config.n_layer, config.layer_kwargs["order"], 1, config.d_model, 1, 1))
            zero_hiddens_init_model_state = None
        elif config.layer == "Mamba_operator": # B * layers * 1 (time) * d_inner * d_state
            config.layer_kwargs["d_inner"] = config.d_model * config.layer_kwargs["expand"]

            zero_hiddens = jax.numpy.zeros((batch_size_per_device, config.n_layer, 1, config.layer_kwargs["d_inner"], config.layer_kwargs["d_state"]))
            zero_hiddens_init_model_state = jax.numpy.zeros((batch_size_per_device // num_devices, config.n_layer, 1, config.layer_kwargs["d_inner"], config.layer_kwargs["d_state"]))
        else:
            raise NotImplementedError(f"Hidden state initialization for {config.layer} not implemented")


        if config.dataset in ["wikitext103"]:
            p_eval_step = jax.pmap(partial(eval_step, config=config, vocab_size=config.vocab_size), axis_name='batch')
        elif config.dataset in ["icl_synthetics"]:
            pass
            # p_eval_step = jax.pmap(partial(eval_step_synthetic, config=config, vocab_size=config.vocab_size), axis_name='batch')
        else:
            raise NotImplementedError("Dataset not implemented")


        rngs, avg_loss, avg_perplexity, avg_accuracy = validate(config, iteration, state, zero_hiddens, val_loader, rngs, val=1, seq_len=seq_len, p_eval_step=p_eval_step)
        evaluation.append((seq_len, 'val', avg_loss, avg_perplexity, avg_accuracy))

        rngs, avg_loss, avg_perplexity, avg_accuracy = validate(config, iteration, state, zero_hiddens, test_loader, rngs, val=2, seq_len=seq_len, p_eval_step=p_eval_step)
        evaluation.append((seq_len, 'test', avg_loss, avg_perplexity, avg_accuracy))

        if evaluate_train:
            rngs, avg_loss, avg_perplexity, avg_accuracy = validate(config, iteration, state, zero_hiddens, train_loader, rngs, val=0, seq_len=seq_len, p_eval_step=p_eval_step)
            evaluation.append((seq_len, 'train', avg_loss, avg_perplexity, avg_accuracy))

    # Write the validation losses to a CSV file
    with open(osp.join(config.output_dir, f'evaluation_consecutive{consecutive_loader}.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Validation sequence Length', 'Dataset Type', 'Loss', 'Perplexity', 'Accuracy'])
        writer.writerows(evaluation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

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

    main()
