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

    # Reshape to (num_devices, device_batch_size, seq_len, dim)
    num_devices = jax.local_device_count()
    inputs = reshape_batch_per_device(inputs, num_devices)
    targets = reshape_batch_per_device(targets, num_devices)
    batch = (inputs, targets)  # Just want to use 1 device batch for init

    batch = get_first_device(batch)
    model = get_model(config)
    state, schedule_fn = init_model_state(init_rng, model, batch[0], config)
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

    # Initialize a list to store validation losses
    validation_losses = []

    seq_len_list = [16 * 2 ** i for i in range(0,12)] # 16 to 32768
    for seq_len in seq_len_list:
        # Change the sequence length for evaluation
        # TODO, care about the out of memory issue. 
        config.l_max = seq_len
        config.data_kwargs["batch_size"] = max(16 * 1024 // seq_len, num_devices)
        config.data_kwargs["batch_size_eval"] = max(16 * 1024 // seq_len, num_devices)
        
        train_loader, val_loader, test_loader = create_wikitext_dataset(config)

        val_loss = validate(config, iteration, state, zero_hiddens, val_loader, rngs, val=1, seq_len=seq_len)
        validation_losses.append((seq_len, 'val', val_loss))

        test_loss = validate(config, iteration, state, zero_hiddens, test_loader, rngs, val=2, seq_len=seq_len)
        validation_losses.append((seq_len, 'test', test_loss))

        train_loss = validate(config, iteration, state, zero_hiddens, train_loader, rngs, val=0, seq_len=seq_len)
        validation_losses.append((seq_len, 'train', train_loss))

    # Write the validation losses to a CSV file
    with open('validation_losses.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Validation sequence Length', 'Dataset Type', 'Loss'])
        writer.writerows(validation_losses)

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
