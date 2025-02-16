import datetime
import os

import wandb
from torch.optim import AdamW
from torch_geometric.data import DataLoader
from tqdm import tqdm

from models.visnet.visnet import ViSNet
from wrappers.BaseModel import BaseModel
import torch
from models.visnet import visnet
from torch.nn import functional as F
import numpy as np
from evaluation.prediction_eval import calculate_metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss, l1_loss
import time




class ViSNetModel(BaseModel):
    def __init__(self, model_name, dataset_name, device=torch.device('cuda'), root_path='../'):
        super().__init__(model_name, dataset_name, device, root_path)
        self.ckpt = None

    def load_model(self, checkpoint_path=None):
        output_dim_map = {
            "all": self.config['model']['out_channels'],
            "young": 3,
            "shear": 3,
            "poisson": 6,
            'density':1,
        }
        self.config['model']['out_channels'] = output_dim_map[self.config['pred_property']]
        self.config['training']['best_model_path'] = os.path.join(self.config['training']['save_dir'], self.config['training']['best_model_name'])
        if not os.path.exists(self.config['training']['save_dir']):
            os.makedirs(self.config['training']['save_dir'])

        self.model = ViSNet(**self.config['model'])
        self.model = self.model.to(self.device)
        if checkpoint_path is not None:
            print('Load checkpoint from', checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.ckpt = ckpt
            self.model.load_state_dict(ckpt['model_state_dict'])


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

        optimizer = AdamW(self.model.parameters(), lr=self.config['training']['lr'], weight_decay=self.config['training']['weight_decay'])
        if hasattr(self, 'ckpt') and self.ckpt is not None:
            optimizer.load_state_dict(self.ckpt['optimizer_state_dict'])
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=self.config['training']['lr_factor'], patience=self.config['training']['lr_patience'], min_lr=self.config['training']['lr_min']
        )


        if self.config['wandb_args']['use_wandb'] == True:
            use_wandb = True
            wandb.init(
                entity=self.config['wandb_args']['entity'],
                project=self.config['wandb_args']['project'],
                name=self.config['wandb_args']['save_name']+'-'+datetime.datetime.now().strftime('%Y-%m-%d--%H:%M'),
                reinit = True
            )
        else:
            use_wandb = False

        best_val_loss = float("inf")
        start_epoch = 0 if self.ckpt is None else self.ckpt['epoch']
        mean_time= []
        bad_count=0
        for epoch in range(start_epoch, self.config['training']['epochs']):
            print(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            # Training loop
            self.model.train()
            train_losses = []
            mean_time_batch = 0
            for batch in train_loader:
                time0 = time.time()
                optimizer.zero_grad()
                batch = batch.to(self.device)
                pred, deriv = self.model(batch)

                loss_fn = mse_loss if self.config['training']['loss_type'] == 'MSE' else l1_loss
                if self.config['pred_property'] == 'young':
                    true_y = batch.y[:,:3]
                elif self.config['pred_property'] == 'shear':
                    true_y = batch.y[:,3:6]
                elif self.config['pred_property'] == 'poisson':
                    true_y = batch.y[:,6:]
                else:
                    true_y = batch.y

                loss_y = loss_fn(pred, true_y)
                loss_dy = loss_fn(deriv, batch['dy']) if self.config['model']['derivative'] and 'dy' in batch else 0

                loss = loss_y * self.config['training']['energy_weight'] + loss_dy * self.config['training']['force_weight']
                loss.backward()
                optimizer.step()
                time1 = time.time()
                mean_time_batch += time1 - time0
                # print(f"Batch time: {time1-time0:.4f}")
            mean_time.append(mean_time_batch / len(train_loader))
            if use_wandb == True:
                wandb.log({"Loss/train": loss.item()})
            train_losses.append(loss.item())

            print(f"Train Loss: {sum(train_losses) / len(train_losses):.4f}, mean_time/batch={np.mean(mean_time)}")

            # Validation loop
            if epoch % self.config['training']['val_check_interval'] == 0 or epoch == self.config['training']['epochs']-1:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        pred, deriv = self.model(batch)
                        if self.config['pred_property'] == 'young':
                            true_y = batch.y[:, :3]
                        elif self.config['pred_property'] == 'shear':
                            true_y = batch.y[:, 3:6]
                        elif self.config['pred_property'] == 'poisson':
                            true_y = batch.y[:, 6:]
                        else:
                            true_y = batch.y
                        loss_y = mse_loss(pred, true_y)
                        loss_dy = 0

                        loss = loss_y * self.config['training']['energy_weight'] + loss_dy * self.config['training']['force_weight']
                        val_losses.append(loss.item())

                val_loss = sum(val_losses) / len(val_losses)

                if use_wandb:
                    wandb.log({"Loss/val": val_loss})
                print(f"Validation Loss: {val_loss:.4f}")

                # Scheduler step
                scheduler.step(val_loss)

                # Save the best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print('Saving checkpoint:', self.config['training']['best_model_path'])

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                    }, self.config['training']['best_model_path'])
                else:
                    bad_count += 1
            if bad_count >= self.config['training']['patience']:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        pred, deriv = self.model(batch)
                        if self.config['pred_property'] == 'young':
                            true_y = batch.y[:, :3]
                        elif self.config['pred_property'] == 'shear':
                            true_y = batch.y[:, 3:6]
                        elif self.config['pred_property'] == 'poisson':
                            true_y = batch.y[:, 6:]
                        else:
                            true_y = batch.y
                        loss_y = mse_loss(pred, true_y)
                        loss_dy = 0

                        loss = loss_y * self.config['training']['energy_weight'] + loss_dy * \
                               self.config['training']['force_weight']
                        val_losses.append(loss.item())

                val_loss = sum(val_losses) / len(val_losses)

                if use_wandb:
                    wandb.log({"Loss/val": val_loss})
                print(f"Validation Loss: {val_loss:.4f}")

                # Scheduler step
                scheduler.step(val_loss)

                # Save the best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print('Saving checkpoint:', self.config['training']['best_model_path'])

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                    }, self.config['training']['best_model_path'])
                break

        print(f'best_val={best_val_loss}, mean_time={np.mean(mean_time)}')
        return best_val_loss, np.mean(mean_time)


    def test(self, return_time=False):
        print("Testing...")
        # self.model.load_state_dict(torch.load(os.path.join(self.config['training']['save_dir'], self.config['training']['best_model_name']), map_location=self.device)['model_state_dict'])
        self.model.eval()
        test_loader = DataLoader(
            dataset=self.test_data,
            batch_size=self.config['training']['valid_batch_size'],
            shuffle=False,
        )

        test_losses = []
        inference_results = {"y_pred": [], "y_true": [], "dy_pred": [], "dy_true": []}
        mean_time_batch = 0
        with torch.no_grad():
            for batch in tqdm(test_loader):
                batch = batch.to(self.device)
                time_start = time.time()
                pred, deriv = self.model(batch)
                time_end = time.time()
                mean_time_batch += time_end - time_start

                if self.config['pred_property'] == 'young':
                    true_y = batch.y[:,:3]
                elif self.config['pred_property'] == 'shear':
                    true_y = batch.y[:,3:6]
                elif self.config['pred_property'] == 'poisson':
                    true_y = batch.y[:,6:]
                else:
                    true_y = batch.y

                inference_results["y_pred"].append(pred.cpu())
                inference_results["y_true"].append(true_y.cpu())
                if self.config['model']['derivative']:
                    inference_results["dy_pred"].append(deriv.cpu())
                    inference_results["dy_true"].append(batch['dy'].cpu())


                loss_y = l1_loss(pred,true_y)
                loss_dy = l1_loss(deriv, batch['dy']) if self.config['model']['derivative'] and 'dy' in batch else 0

                loss = loss_y * self.config['training']['energy_weight'] + loss_dy * self.config['training']['force_weight']
                test_losses.append(loss.item())
            mean_time_batch /= len(test_loader)
        test_loss = sum(test_losses) / len(test_losses)
        print(f"Test Loss: {test_loss:.4f}, mean_time/batch={mean_time_batch}")



        pred_all = torch.cat(inference_results['y_pred'], dim=0).cpu().numpy()
        targe_all = torch.cat(inference_results['y_true'], dim=0).cpu().numpy()
        r2, nrmse, mae = calculate_metrics(pred_all, targe_all)
        if self.config['wandb_args']['use_wandb'] and wandb.run:
            wandb.log({"Test/R2": r2, 'NRMSE': nrmse, 'MAE': mae})
        print(f'R2:{r2}, NRMSE:{nrmse}, MAE:{mae}')
        if return_time:
            return r2, nrmse, mae, mean_time_batch
        else:
            return r2, nrmse, mae



    def save_results(self, path='results/'):
        print("Saving results...")
        # Implement saving logic for inference results or metrics here