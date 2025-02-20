from argparse import Namespace

from models.mace_ve.lattices.utils import elasticity_func
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

class MaceVeModel(BaseModel):

    def evaluate(self):
        pass

    def test(self):
        pass


    def load_model(self):
        max_edge_radius = self.train_data.edge_feat.max()
        self.config['network']['max_edge_radius'] = max_edge_radius
        self.model = EnergyEquivGNN(params=Namespace(**self.config['network']))
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
            betas=(params.beta1,0.999), eps=params.epsilon,
            amsgrad=params.amsgrad, weight_decay=params.weight_decay,)

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
        writer = SummaryWriter(log_dir=params.log_dir)  # TensorBoard logger
        checkpoint_manager = CheckpointManager(model=self.model, save_dir=params.log_dir)  # Assuming the second callback is CheckpointManager
        early_stopper = EarlyStopper(patience=50, mode='min')  # Assuming the third callback is EarlyStopper

        global_step = 0
        self.model.train()
        last_time_metrics = {'_last_step': 0, '_last_time': time.time()}

        while global_step < params.max_steps:
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device)
                # Forward pass
                shifts = torch.zeros((batch.edge_index.shape[1], 3), dtype=torch.float32, device=device)
                num_graphs = batch.batch.max().item() + 1
                output = self.model(batch.cart_coords, batch.edge_index, shifts, batch.edge_feat, batch.node_feat, num_graphs, batch.batch)

                true_y = batch.y
                pred_y = output['y']

                target = true_y
                predicted = pred_y
                mean_stiffness = target.pow(2).mean(dim=1)
                stiffness_loss = F.mse_loss(predicted, target, reduction='none').mean(dim=1)
                loss = 100 * (stiffness_loss / mean_stiffness).mean()

                # Backward pass and optimization
                loss.backward()

                if (batch_idx + 1) % params.accumulate_grad_batches == 0:
                    if params.gradient_clip_val is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), params.gradient_clip_val)
                    optimizer.step()
                    optimizer.zero_grad()

                    # Log training metrics
                    if global_step % params.log_every_n_steps == 0:
                        writer.add_scalar('Loss/train', loss.item(), global_step)
                        print(
                            f'Step {global_step}, Loss: {loss.item()}, Stiffness Loss Mean: {stiffness_loss.mean().item()}')

                    # Validation check
                    if global_step % params.val_check_interval == 0:
                        val_loss = self.validate(self.model, val_loader, device, writer, global_step)
                        checkpoint_manager.step({'val_loss': val_loss}, epoch=None, step=global_step, optimizer=optimizer)
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

    def validate(self, model, val_loader, device, writer, global_step):
        model.eval()
        all_val_losses = []
        # all_val_stiff_dir_losses = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                # directions = torch.randn(250, 3, dtype=torch.float32, device=device)
                # directions = directions / directions.norm(dim=-1, keepdim=True)

                output = model(batch)
                true_stiffness = batch.y
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
        writer.add_scalar('Loss/val', avg_val_loss, global_step)
        # writer.add_scalar('Loss/val_stiff_dir', avg_val_stiff_dir_loss, global_step)
        print(f'Validation Loss: {avg_val_loss}. '
              # f'Val Stiff Dir Loss: {avg_val_stiff_dir_loss}'
              )
        return avg_val_loss



class CheckpointManager:
    def __init__(self, model, save_dir, filename='{epoch}-{step}-{val_loss:.3f}', monitor='val_loss', save_top_k=1):
        self.model = model
        self.save_dir = Path(save_dir)
        self.filename = filename
        self.monitor = monitor
        self.save_top_k = save_top_k
        self.best_scores = []
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def step(self, metrics, epoch, step, optimizer):
        score = metrics[self.monitor]
        if len(self.best_scores) < self.save_top_k or score < max(self.best_scores):
            self.best_scores.append(score)
            self.best_scores.sort()
            if len(self.best_scores) > self.save_top_k:
                self.best_scores.pop()

            checkpoint_path = self.save_dir / self.filename.format(epoch=epoch, step=step, val_loss=score)
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': score,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")


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


