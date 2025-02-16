import logging
import os
import time
from argparse import Namespace

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from timm.utils import NativeScaler
from timm.utils import ModelEmaV2
from timm.scheduler import create_scheduler
import wandb
import datetime
from contextlib import suppress

from tqdm import tqdm

from evaluation.prediction_eval import calculate_metrics
from models.euiformer.optim_factory import create_optimizer
from models.euiformer.engine import train_one_epoch, evaluate, compute_stats
from models.euiformer.nets import model_entrypoint
from wrappers.BaseModel import BaseModel

class EquiformerModel(BaseModel):
    def __init__(self, model_name, dataset_name, device=torch.device('cuda'), root_path='../', config=None):
        super().__init__(model_name, dataset_name, device, root_path)
        self.loss_scaler = None
        self.model_ema = None
        self.norm_factor = [0, 1]
        self.target = None

    def load_model(self, checkpoint_path=None):
        output_dim_map = {
            "all": self.config['model']['output_channels'],
            "young": 3,
            "shear": 3,
            "poisson": 6,
            'density': 1,
        }
        target_map = {
            "all": [i for i in range(self.config['model']['output_channels'])],
            "young": [0,1,2],
            "shear": [3,4,5],
            "poisson": [6,7,8,9,10,11],
            'density': [0],
        }
        self.config['model']['output_channels'] = output_dim_map[self.config['pred_property']]
        self.target = target_map[self.config['pred_property']]

        create_model = model_entrypoint(self.config['model']['model_name'])
        self.model = create_model(
            irreps_in=self.config['model']['input_irreps'],
            radius=self.config['model']['radius'],
            num_basis=self.config['model']['num_basis'],
            out_channels=self.config['model']['output_channels'],
            task_mean=self.norm_factor[0],
            task_std=self.norm_factor[1],
            atomref=None,
            drop_path=self.config['model']['drop_path'],
            output_size=str(self.config['model']['output_channels'])+'x0e',
        ).to(self.device)
        if checkpoint_path is not None:
            print('Load checkpoint from', checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.ckpt = ckpt
            self.model.load_state_dict(ckpt['model_state_dict'])


        if self.config['training']['model_ema']:
            self.model_ema = ModelEmaV2(
                self.model,
                decay=self.config['training']['model_ema_decay'],
                device='cpu' if self.config['training']['model_ema_force_cpu'] else None
            )
        return self.model

    def train(self):


        train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
        )
        val_loader = DataLoader(
            dataset=self.val_data,
            batch_size=self.config['training']['valid_batch_size'],
            shuffle=False,
        )

        optimizer = create_optimizer(Namespace(**self.config['training']), self.model)
        lr_scheduler, _ = create_scheduler(Namespace(**self.config['training']), optimizer)

        if self.config['task']['loss'] == 'l1':
            criterion = torch.nn.L1Loss()
        elif self.config['task']['loss'] == 'l2':
            criterion = torch.nn.MSELoss()
        else:
            raise ValueError("Invalid loss type")

        amp_autocast = suppress
        if self.config['amp']['enabled']:
            amp_autocast = torch.cuda.amp.autocast
            self.loss_scaler = NativeScaler()

        if self.config['wandb_args']['use_wandb'] == True:
            use_wandb = True
            wandb.init(
                entity=self.config['wandb_args']['entity'],
                project=self.config['wandb_args']['project'],
                name=self.config['wandb_args']['save_name'] + '-' + datetime.datetime.now().strftime('%Y-%m-%d--%H:%M'),
                reinit = True
            )
        else:
            use_wandb = False
        os.makedirs(self.config['log_dir'], exist_ok=True)
        logging.basicConfig(filename=self.config['log_dir']+f'/{self.config["pred_property"]}.log')
        logger = logging.getLogger('logger')
        self.amp_autocast = amp_autocast = torch.cuda.amp.autocast if self.config['amp']['enabled'] else suppress
        best_metrics = {
            'epoch': 0,
            'train_err': float('inf'),
            'val_err': float('inf'),
            'test_err': float('inf'),
            'r2': -float('inf'),
            'nrmse': float('inf'),
            'mae': float('inf')
        }
        bad_count = 0
        mean_time = 0
        for epoch in range(self.config['training']['epochs']):
            lr_scheduler.step(epoch)

            train_err, mean_time_per_batch = train_one_epoch(
                model=self.model,
                criterion=criterion,
                norm_factor=self.norm_factor,
                target=self.target,
                data_loader=train_loader,
                optimizer=optimizer,
                device=self.device,
                epoch=epoch,
                model_ema=self.model_ema,
                amp_autocast=amp_autocast,
                loss_scaler=self.loss_scaler,
                print_freq=self.config['logging']['print_freq'],
                wandb_args = self.config['wandb_args'],
                logger=logger
            )
            mean_time += mean_time_per_batch
            print(f'Epoch: {epoch}, mean_time:{mean_time / (epoch+1)}')
            if use_wandb == True:
                wandb.log({"Loss/train": train_err})
            if epoch % self.config['training']['valid_freq'] == 0 or epoch == self.config['training']['epochs'] - 1:
                val_err, _, r2, nrmse, mae = evaluate(self.model, self.norm_factor, self.target, val_loader, self.device, amp_autocast)
                if use_wandb == True:
                    wandb.log(best_metrics)
                    wandb.log({"Loss/val": val_err})

                if r2 >= best_metrics['r2']:
                    best_metrics.update({
                        'best_epoch': epoch,
                        'train_err': train_err,
                        'val_err': val_err,
                        'r2': r2,
                        'nrmse':nrmse,
                        'mae':mae
                    })
                    print('Saving results...')
                    self.save_results(self.config['output_dir'])
                else:
                    bad_count += 1

            if bad_count >= self.config['training']['patience']:
                val_err, _, r2, nrmse, mae = evaluate(self.model, self.norm_factor, self.target, val_loader,
                                                      self.device, amp_autocast)
                if use_wandb == True:
                    wandb.log(best_metrics)
                    wandb.log({"Loss/val": val_err})

                if r2 >= best_metrics['r2']:
                    best_metrics.update({
                        'best_epoch': epoch,
                        'train_err': train_err,
                        'val_err': val_err,
                        'r2': r2,
                        'nrmse': nrmse,
                        'mae': mae
                    })
                    print('Saving results...')
                    self.save_results(self.config['output_dir'])
                print(f"Epoch {epoch}: Train Error: {train_err}, Val Error: {val_err}", best_metrics)

                break

    def test(self):
        print("Testing...")
        self.model.load_state_dict(torch.load(self.config['output_dir']+'/best_model.pth', map_location=self.device)['model_state_dict'])
        self.model.eval()
        test_loader = DataLoader(
            dataset=self.test_data,
            batch_size=self.config['training']['valid_batch_size'],
            shuffle=False,
        )

        pred_all = []
        targe_all = []
        mean_time = 0
        with torch.no_grad():
            for data in tqdm(test_loader):
                data = data.to(self.device)
                start_time = time.time()
                # data.edge_d_index = radius_graph(data.pos, r=10.0, batch=data.batch, loop=True)
                # data.edge_d_attr = data.edge_attr
                pred_y = self.model(f_in=data.node_feat, pos=data.cart_coords, batch=data.batch,
                             node_atom=data.node_type,
                             edge_d_index=data.edge_index, edge_d_attr=data.edge_feat)
                end_time = time.time()
                mean_time += (end_time - start_time)
                true_y = data.y[:, self.target]
            pred_all.append(pred_y)
            targe_all.append(true_y)
        pred_all = torch.cat(pred_all, dim=0).cpu().numpy()
        targe_all = torch.cat(targe_all, dim=0).cpu().numpy()
        r2, nrmse, mae = calculate_metrics(pred_all, targe_all)
        if self.config['wandb_args']['use_wandb'] and wandb.run:
            wandb.log({"Test/R2": r2, 'NRMSE': nrmse, 'MAE': mae})
        print(f'R2:{r2}, NRMSE:{nrmse}, MAE:{mae}, mean_time:{mean_time / len(test_loader)}')

        return r2, nrmse, mae

    def evaluate(self):
        test_loader = DataLoader(
            dataset=self.test_data,
            batch_size=self.config['training']['valid_batch_size'],
            shuffle=False,
        )
        val_loader = DataLoader(
            dataset=self.val_data,
            batch_size=self.config['training']['valid_batch_size'],
            shuffle=False,
        )
        val_err, _, r2, nrmse, mae = evaluate(self.model, self.norm_factor, self.target, val_loader, self.device, self.amp_autocast)
        test_err, _, r2, nrmse, mae = evaluate(self.model, self.norm_factor, self.target, test_loader, self.device, self.amp_autocast)
        print(f"Validation Error: {val_err}, Test Error: {test_err}")

    def save_results(self, path='results/'):
        os.makedirs(path, exist_ok=True)
        results = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }
        torch.save(results, os.path.join(path, 'best_model.pth'))

    def visualize(self, path='results/plots/'):
        pass  # Implement visualization logic here

