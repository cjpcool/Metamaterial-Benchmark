from argparse import Namespace
import datetime

import wandb
from tqdm import tqdm

from wrappers.BaseModel import BaseModel
import torch
from models.mace_ve import EnergyEquivGNN
import time
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
from torch.nn import functional as F
import numpy as np
from evaluation.prediction_eval import calculate_metrics

class MaceVeModel(BaseModel):

    def __init__(self, model_name, dataset_name, device=torch.device('cuda'), root_path='../'):
        super(MaceVeModel, self).__init__(model_name, dataset_name, device, root_path)
        self.ckpt = None

    def evaluate(self):
        print('Evaluating...')
        params = Namespace(**self.config['training'])
        if self.device is None:
            device = torch.device("cuda" if params.use_cuda and torch.cuda.is_available() else "cpu")
        else:
            device = self.device
        self.model = self.model.to(device)

        val_loader = DataLoader(
            dataset=self.val_data,
            batch_size=params.valid_batch_size,
            shuffle=False,
            num_workers=params.num_workers,
        )

        pred_all = []
        targe_all = []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                batch = batch.to(device)
                # directions = torch.randn(250, 3, dtype=torch.float32, device=device)
                # directions = directions / directions.norm(dim=-1, keepdim=True)

                shifts = torch.zeros((batch.edge_index.shape[1], 3), dtype=torch.float32, device=device)
                num_graphs = batch.batch.max().item() + 1
                output = self.model(batch.cart_coords, batch.edge_index, shifts, batch.edge_feat, batch.node_feat,
                                    num_graphs, batch.batch)
                if params.pred_property == 'young':
                    true_y = batch.y[:,:3]
                elif params.pred_property == 'shear':
                    true_y = batch.y[:,3:6]
                elif params.pred_property == 'poisson':
                    true_y = batch.y[:,6:]
                else:
                    true_y = batch.y
                pred_y = output['y']

                pred_all.append(pred_y)
                targe_all.append(true_y)
        pred_all = torch.cat(pred_all, dim=0).cpu().numpy()
        targe_all = torch.cat(targe_all, dim=0).cpu().numpy()
        r2, nrmse, mae = calculate_metrics(pred_all, targe_all)
        if self.config['wandb_args']['use_wandb'] and wandb.run:
            wandb.log({"Eval/R2": r2, 'NRMSE': nrmse, 'MAE': mae})

        print(f'R2:{r2}, NRMSE:{nrmse}, MAE:{mae}')
        return r2, nrmse, mae

    def evaluate_train(self):
        print('Evaluating...')
        params = Namespace(**self.config['training'])
        if self.device is None:
            device = torch.device("cuda" if params.use_cuda and torch.cuda.is_available() else "cpu")
        else:
            device = self.device
        self.model = self.model.to(device)

        train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=params.valid_batch_size,
            shuffle=False,
            num_workers=params.num_workers,
        )

        pred_all = []
        targe_all = []
        with torch.no_grad():
            for batch in tqdm(train_loader):
                batch = batch.to(device)
                # directions = torch.randn(250, 3, dtype=torch.float32, device=device)
                # directions = directions / directions.norm(dim=-1, keepdim=True)

                shifts = torch.zeros((batch.edge_index.shape[1], 3), dtype=torch.float32, device=device)
                num_graphs = batch.batch.max().item() + 1
                output = self.model(batch.cart_coords, batch.edge_index, shifts, batch.edge_feat, batch.node_feat,
                                    num_graphs, batch.batch)
                if params.pred_property == 'young':
                    true_y = batch.y[:,:3]
                elif params.pred_property == 'shear':
                    true_y = batch.y[:,3:6]
                elif params.pred_property == 'poisson':
                    true_y = batch.y[:,6:]
                else:
                    true_y = batch.y
                pred_y = output['y']

                pred_all.append(pred_y)
                targe_all.append(true_y)
        pred_all = torch.cat(pred_all, dim=0).cpu().numpy()
        targe_all = torch.cat(targe_all, dim=0).cpu().numpy()
        r2, nrmse, mae = calculate_metrics(pred_all, targe_all)
        if self.config['wandb_args']['use_wandb'] and wandb.run:
            wandb.log({"Eval_train/R2": r2.item(), 'NRMSE': nrmse.item(), 'MAE': mae.item()})
        print(f'R2:{r2}, NRMSE:{nrmse}, MAE:{mae}')
        return r2, nrmse, mae

    def test(self):
        print('Testing...')
        params = Namespace(**self.config['training'])
        if self.device is None:
            device = torch.device("cuda" if params.use_cuda and torch.cuda.is_available() else "cpu")
        else:
            device = self.device
        self.model = self.model.to(device)


        test_loader = DataLoader(
            dataset=self.test_data,
            batch_size=params.valid_batch_size,
            shuffle=False,
            num_workers=params.num_workers,
        )

        pred_all = []
        targe_all = []
        mean_time= 0
        with torch.no_grad():
            for batch in tqdm(test_loader):
                batch = batch.to(device)
                start_time = time.time()
                # directions = torch.randn(250, 3, dtype=torch.float32, device=device)
                # directions = directions / directions.norm(dim=-1, keepdim=True)

                shifts = torch.zeros((batch.edge_index.shape[1], 3), dtype=torch.float32, device=device)
                num_graphs = batch.batch.max().item() + 1
                output = self.model(batch.cart_coords, batch.edge_index, shifts, batch.edge_feat, batch.node_feat,
                                    num_graphs, batch.batch)
                end_time = time.time()
                mean_time += (end_time - start_time)
                if params.pred_property == 'young':
                    true_y = batch.y[:,:3]
                elif params.pred_property == 'shear':
                    true_y = batch.y[:,3:6]
                elif params.pred_property == 'poisson':
                    true_y = batch.y[:,6:]
                else:
                    true_y = batch.y
                pred_y = output['y']

                pred_all.append(pred_y)
                targe_all.append(true_y)
        pred_all = torch.cat(pred_all, dim=0).cpu().numpy()
        targe_all = torch.cat(targe_all, dim=0).cpu().numpy()
        r2, nrmse, mae = calculate_metrics(pred_all, targe_all)
        if self.config['wandb_args']['use_wandb'] and wandb.run:
            wandb.log({"Test/R2": r2, 'NRMSE': nrmse, 'MAE': mae})
        print(f'R2:{r2}, NRMSE:{nrmse}, MAE:{mae}, mean_time:{mean_time / len(test_loader)}')
        return r2, nrmse, mae


    def load_model(self, checkpoint_path=None):
        max_edge_radius = self.train_data.edge_feat.max()
        self.config['network']['max_edge_radius'] = max_edge_radius
        output_dim_map = {
            "all": self.config['network']['output_dim'],
            "young": 3,
            "shear": 3,
            "poisson": 6,
            'density': 1,
        }
        self.config['network']['output_dim'] = output_dim_map[self.config['training']['pred_property']]

        self.model = EnergyEquivGNN(params=Namespace(**self.config['network']))
        if checkpoint_path is not None:
            print('Load checkpoint from', checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.ckpt = ckpt
            self.model.load_state_dict(ckpt['model_state_dict'])
        return self.model

    def save_results(self, path='results/'):
        pass


    def train(self):
        '''
        output = self.model(batch)

        true_stiffness = batch['stiffness']
        pred_stiffness = output['stiffness']

        target = true_stiffness  # [N, 6, 6]
        predicted = pred_stiffness  # [N, 6, 6]
        mean_stiffness = target.pow(2).mean(dim=(1, 2))  # [N]
        stiffness_loss = torch.nn.functional.mse_loss(predicted, target, reduction='none').mean(dim=(1, 2))  # [N]

        stiffness_loss_mean = stiffness_loss.mean()

        loss = stiffness_loss
        loss = 100 * (loss / mean_stiffness).mean()  # [1]
        :return:
        '''

        params = Namespace(**self.config['training'])

        if self.device is None:
            device = torch.device("cuda" if params.use_cuda and torch.cuda.is_available() else "cpu")
        else:
            device = self.device
        self.model = self.model.to(device)

        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=params.lr,
                                      betas=(params.beta1, 0.999), eps=params.epsilon,
                                      amsgrad=params.amsgrad, weight_decay=params.weight_decay, )
        if hasattr(self, 'ckpt') and self.ckpt is not None:
            optimizer.load_state_dict(self.ckpt['optimizer_state_dict'])

        train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=params.num_workers,
        )
        val_loader = DataLoader(
            dataset=self.val_data,
            batch_size=params.valid_batch_size,
            shuffle=False,
            num_workers=params.num_workers,
        )
        if self.config['wandb_args']['use_wandb'] == True:
            use_wandb = True
            wandb.init(
                entity=self.config['wandb_args']['entity'],
                project=self.config['wandb_args']['project'],
                name=self.config['wandb_args']['save_name']+'-'+datetime.datetime.now().strftime('%Y-%m-%d--%H:%M'),
                reinit=True
            )
            writer=None
        else:
            use_wandb=False
            writer = SummaryWriter(log_dir=params.log_dir)  # TensorBoard logger
        checkpoint_manager = CheckpointManager(model=self.model, save_dir=params.save_dir)  # Assuming the second callback is CheckpointManager
        early_stopper = EarlyStopper(patience=self.config['training']['patience'], mode='min')  # Assuming the third callback is EarlyStopper

        global_step = 0
        self.model.train()
        last_time_metrics = {'_last_step': 0, '_last_time': time.time()}
        epoch = 0 if self.ckpt is None else self.ckpt['epoch']
        while global_step < params.max_steps:
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device)
                # Forward pass
                shifts = torch.zeros((batch.edge_index.shape[1], 3), dtype=torch.float32, device=device)
                num_graphs = batch.batch.max().item() + 1

                output = self.model(batch.cart_coords, batch.edge_index, shifts, batch.edge_feat, batch.node_feat, num_graphs, batch.batch)

                if params.pred_property == 'young':
                    true_y = batch.y[:,:3]
                elif params.pred_property == 'shear':
                    true_y = batch.y[:,3:6]
                elif params.pred_property == 'poisson':
                    true_y = batch.y[:,6:]
                else:
                    true_y = batch.y
                pred_y = output['y']

                target = true_y
                predicted = pred_y
                # mean_stiffness = target.pow(2).mean(dim=1)
                stiffness_loss = F.mse_loss(predicted, target, reduction='none').mean(dim=1)
                # loss = 100 * (stiffness_loss / mean_stiffness).mean()
                loss = 100 * stiffness_loss.mean()
                # Backward pass and optimization
                loss.backward()

                if (batch_idx + 1) % params.accumulate_grad_batches == 0:
                    if params.gradient_clip_val is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), params.gradient_clip_val)
                    optimizer.step()
                    optimizer.zero_grad()

                    # Log training metrics
                    if global_step % params.log_every_n_steps == 0:
                        if use_wandb:
                            wandb.log({"Loss/train": loss.item()}, step=global_step)
                        else:
                            writer.add_scalar('Loss/train', loss.item(), global_step)
                        print(
                            f'Step {global_step}, Loss: {loss.item()}, Loss Mean: {stiffness_loss.mean().item()}')

                    # Validation check
                    if global_step % params.val_check_interval == 0:
                        val_loss = self.validate(self.model, val_loader, params, device, writer, global_step, use_wandb)
                        checkpoint_manager.step({'val_loss': val_loss}, epoch=epoch, step=global_step, optimizer=optimizer)
                        early_stopper.step(val_loss)
                        self.model.train()  # Return to training mode

                    # Time metrics logging
                    steps_done = global_step - last_time_metrics['_last_step']
                    time_now = time.time()
                    time_taken = time_now - last_time_metrics['_last_time']
                    steps_per_sec = steps_done / time_taken
                    last_time_metrics['_last_step'] = global_step
                    last_time_metrics['_last_time'] = time_now
                    print(f'Steps per second: {steps_per_sec}')

                    # Check for NaN loss
                    if torch.isnan(loss):
                        print('Loss is NaN. Stopping training.')
                        return

                    global_step += 1

                    if global_step >= params.max_steps or early_stopper.should_stop:
                        break
            epoch += 1

    def validate(self, model, val_loader, params, device, writer, global_step, use_wandb):
        model.eval()
        all_val_losses = []
        # all_val_stiff_dir_losses = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                # directions = torch.randn(250, 3, dtype=torch.float32, device=device)
                # directions = directions / directions.norm(dim=-1, keepdim=True)

                shifts = torch.zeros((batch.edge_index.shape[1], 3), dtype=torch.float32, device=device)
                num_graphs = batch.batch.max().item() + 1
                output = self.model(batch.cart_coords, batch.edge_index, shifts, batch.edge_feat, batch.node_feat,
                                    num_graphs, batch.batch)
                if params.pred_property == 'young':
                    true_y = batch.y[:,:3]
                elif params.pred_property == 'shear':
                    true_y = batch.y[:,3:6]
                elif params.pred_property == 'poisson':
                    true_y = batch.y[:,6:]
                else:
                    true_y = batch.y
                true_stiffness = true_y
                pred_stiffness = output['y']

                stiffness_loss = F.mse_loss(pred_stiffness, true_stiffness)

                # true_stiff_4 = elasticity_func.stiffness_Mandel_to_cart_4(true_stiffness)
                # pred_stiff_4 = elasticity_func.stiffness_Mandel_to_cart_4(pred_stiffness)
                # stiff_dir_true = torch.einsum('...ijkl,pi,pj,pk,pl->...p', true_stiff_4, directions, directions,
                #                               directions, directions)
                # stiff_dir_pred = torch.einsum('...ijkl,pi,pj,pk,pl->...p', pred_stiff_4, directions, directions,
                #                               directions, directions)
                # stiff_dir_loss = F.l1_loss(stiff_dir_pred, stiff_dir_true)

                all_val_losses.append(stiffness_loss.item())
                # all_val_stiff_dir_losses.append(stiff_dir_loss.item())

        avg_val_loss = np.mean(all_val_losses)
        # avg_val_stiff_dir_loss = np.mean(all_val_stiff_dir_losses)
        if use_wandb:
            wandb.log({"Loss/val": avg_val_loss.item()}, step=global_step)
        else:
            writer.add_scalar('Loss/val', avg_val_loss, global_step)
            # writer.add_scalar('Loss/val_stiff_dir', avg_val_stiff_dir_loss, global_step)
        print(f'Validation Loss: {avg_val_loss}. '
              # f'Val Stiff Dir Loss: {avg_val_stiff_dir_loss}'
              )
        return avg_val_loss


class CheckpointManager:
    def __init__(self, model, save_dir, filename='{epoch}-{step}-{val_loss:.5f}.pth', monitor='val_loss', save_top_k=5):
        self.model = model
        self.save_dir = Path(save_dir)
        self.filename = filename
        self.monitor = monitor
        self.save_top_k = save_top_k
        self.best_scores = []
        self.checkpoints = []
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def step(self, metrics, epoch, step, optimizer):
        score = metrics[self.monitor]

        # Check if we should save the checkpoint
        if len(self.best_scores) < self.save_top_k or score < max(self.best_scores):
            # Add the new score and sort the best scores
            self.best_scores.append(score)
            self.best_scores.sort()
            if len(self.best_scores) > self.save_top_k:
                self.best_scores.pop()

            # Save the new checkpoint
            checkpoint_path = self.save_dir / self.filename.format(epoch=epoch, step=step, val_loss=score)
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': score,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # Add the new checkpoint to the list
            self.checkpoints.append((checkpoint_path, score))

            # Save the best model separately
            self._save_best_model()

            # Remove checkpoints not in the top-k
            self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        # Keep only the top-k checkpoints
        self.checkpoints.sort(key=lambda x: x[1])  # Sort by score (ascending)
        while len(self.checkpoints) > self.save_top_k:
            checkpoint_to_remove, _ = self.checkpoints.pop()  # Remove the worst checkpoint
            if checkpoint_to_remove.exists():
                os.remove(checkpoint_to_remove)
                print(f"Removed checkpoint: {checkpoint_to_remove}")

    def _save_best_model(self):
        if self.best_scores:
            best_score = self.best_scores[0]
            for checkpoint_path, score in self.checkpoints:
                if score == best_score:
                    best_model_path = self.save_dir / 'best_model.pth'
                    torch.save(torch.load(checkpoint_path), best_model_path)
                    print(f"Saved best model: {best_model_path}")
                    break



class EarlyStopper:
    def __init__(self, patience=50, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'min' and score > self.best_score - self.min_delta) or \
                (self.mode == 'max' and score < self.best_score + self.min_delta):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0


