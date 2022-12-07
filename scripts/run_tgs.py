###### for PERCH Desktop only #######
import sys
sys.path.append('../')

import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
from typing import List

def random_split(dataset, lengths,
                 generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]
###### for PERCH Desktop only #######

import torch
from torch import nn
from torch.utils.data import DataLoader#, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


from data.tgs_salt import SaltDataset

from argparse import ArgumentParser

from models.segmentor import BaseSegmentor

def main():
    parser = ArgumentParser()

    # wandb logger args
    parser.add_argument("--project", type=str, default="cis520")
    parser.add_argument("--entity", type=str, default="mlfp")
    parser.add_argument("--group", type=str, default="salt")
    parser.add_argument("--name", type=str, default="test")

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    parser = BaseSegmentor.add_model_specific_args(parser)

    # data args
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset_split", type=float, default=0.7)
    parser.add_argument("--seed", type=float, default=42)

    parser.add_argument("--num_workers", type=int, default=12)
    # position embedding
    parser.add_argument("--pos_embed", action='store_true')
    parser.add_argument("--embed_dim", type=int, default=16)
    # use y mean as scalar?
    parser.add_argument("--use_ymean", action='store_true')

    args = parser.parse_args()
    dict_args = vars(args)
    wandb_logger = WandbLogger(project = args.project, entity = args.entity, group = args.group, name = args.name)
    
    # get dataset
    dataset = SaltDataset()
    # set random seed
    seed = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [args.dataset_split, 1 - args.dataset_split], generator=seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)
    # trainer = pl.Trainer.from_argparse_args(args) # for PERCH local test

    dict_args['meta_dim'] = 1 # for salt dataset we just have scalar depth metadata
    if args.use_ymean:
        dict_args['meta_dim'] = 101 # if use y mean as metadata
    dict_args['pos_embed'] = args.pos_embed
    dict_args['embed_dim'] = args.embed_dim
    model = BaseSegmentor(**dict_args)
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

if __name__ == '__main__':
    # use this command to see tensorboard:
    # tensorboard --logdir=scripts/lightning_logs/
    main()
