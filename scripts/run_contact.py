import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


from data.contact_dataset import ContactDataset

from argparse import ArgumentParser

from models.segmentor import BaseSegmentor

def main():
    parser = ArgumentParser()

    # wandb logger args
    parser.add_argument("--project", type=str, default="cis520")
    parser.add_argument("--entity", type=str, default="mlfp")
    parser.add_argument("--group", type=str, default="contact")
    parser.add_argument("--name", type=str, default="test")

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    parser = BaseSegmentor.add_model_specific_args(parser)

    # data args
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset_split", type=float, default=0.7)
    parser.add_argument("--seed", type=float, default=42)
    parser.add_argument("--num_workers", type=int, default=12)
    
    # position embedding
    parser.add_argument("--pos_embed", action='store_true')
    parser.add_argument("--embed_dim", type=int, default=16)

    args = parser.parse_args()
    dict_args = vars(args)
    wandb_logger = WandbLogger(project = args.project, entity = args.entity, group = args.group, name = args.name)
    
    # get dataset
    dataset = ContactDataset()
    # set random seed
    seed = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [args.dataset_split, 1 - args.dataset_split], generator=seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)
    
    dict_args['meta_dim'] = 13 # 7d pose + 6d wrench
    dict_args['threshold'] = dataset.threshold 
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
