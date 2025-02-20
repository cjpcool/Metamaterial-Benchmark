####################################################################
# Callbacks for pytorch lightning
# Author: Ivan Grega
# This program is distributed under the MIT license
####################################################################

import copy
import logging
from typing import Any, Optional, Dict
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import wandb

class PrintTableMetrics(pl.Callback):
    def __init__(self, log_metrics: list, every_n_steps: int = 1000, col_width: int = 10) -> None:
        super().__init__()

        header = []
        for metric in log_metrics:
            header.append(metric)
        if 'epoch' not in header:
            header.insert(0, "epoch")
        
        self.every_n_steps = every_n_steps
        self.format_str = '{' + ':<' + str(col_width) + '}'
        self.col_width = col_width
        n_cols = len(header)
        total_width = col_width * n_cols + 3*n_cols
        self.total_width = total_width
        
        self.header = header
        self._time_metrics = {}

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        s = self.format_str
        fields = [s.format(metric) for metric in self.header]
        line = " | ".join(fields) + "\n" + "-" * self.total_width
        rank_zero_info(line)
        self._time_metrics['_last_step'] = trainer.global_step
        self._time_metrics['_last_time'] = time.time()

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        line = "\n" + "-" * self.total_width
        rank_zero_info(line)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        metrics_dict = copy.copy(trainer.callback_metrics)
        local_dict = {key: metrics_dict[key].detach().cpu().item() for key in metrics_dict.keys()}
        local_dict['epoch'] = trainer.current_epoch
        rank_zero_info(self._format_table(local_dict))

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: "STEP_OUTPUT", batch: Any, batch_idx: int) -> None:
        if (trainer.global_step%self.every_n_steps)==0:
            metrics_dict = copy.copy(trainer.callback_metrics)
            local_dict = {key: metrics_dict[key].detach().cpu().item() for key in metrics_dict.keys()}
            local_dict['step'] = trainer.global_step
            lrs = [pg['lr'] for pg in pl_module.optimizers().param_groups]
            lr = sum(lrs)/len(lrs)
            local_dict['lr'] = lr
            ##
            tn = time.time()
            step_now = trainer.global_step
            steps_done = step_now - self._time_metrics['_last_step']
            time_elapsed = tn - self._time_metrics['_last_time']
            if steps_done>0:
                step_per_time = steps_done/time_elapsed
                local_dict['steps_per_time'] = step_per_time
                local_dict['samples_per_time'] = step_per_time * batch.num_graphs
            self._time_metrics['_last_time'] = tn
            self._time_metrics['_last_step'] = step_now

            rank_zero_info(self._format_table(local_dict))
        
    
    def _format_table(self, metrics: dict) -> str:
        # Formatting
        s = self.format_str
        fields = []
        for key in self.header:
            if key in metrics:
                if isinstance(metrics[key], float):
                    val = f'{metrics[key]:.6f}'
                else:
                    val = metrics[key]
                fields.append(s.format(val))
            else:
                fields.append(s.format(''))
        line =  " | ".join(fields)
        return line
        

class SimpleTableMetrics:

    def __init__(self, log_metrics: list, every_n_steps: int = 1000, col_width: int = 10) -> None:

        header = []
        for metric in log_metrics:
            header.append(metric)
        if 'epoch' not in header:
            header.insert(0, "epoch")
        
        self.every_n_steps = every_n_steps
        self.format_str = '{' + ':<' + str(col_width) + '}'
        self.col_width = col_width
        n_cols = len(header)
        total_width = col_width * n_cols + 3*n_cols
        self.total_width = total_width
        
        self.header = header
        self._time_metrics = {}
        self.on_fit_start()

    def on_fit_start(self) -> None:
        s = self.format_str
        fields = [s.format(metric) for metric in self.header]
        line = " | ".join(fields) + "\n" + "-" * self.total_width
        rank_zero_info(line)
        self._time_metrics['_last_step'] = 0
        self._time_metrics['_last_time'] = time.time()

    def on_fit_end(self) -> None:
        line = "\n" + "-" * self.total_width
        rank_zero_info(line)

    def on_epoch_start(self, local_dict: Dict) -> None:
        rank_zero_info(self._format_table(local_dict))
        
    def on_validation(self, local_dict: Dict) -> None:
        rank_zero_info(self._format_table(local_dict))

    def on_train_batch_end(self, local_dict: Dict) -> None:
        step_now = local_dict['step']
        if (step_now==0) or ((step_now+1)%self.every_n_steps==0):
            tn = time.time()
            steps_done = step_now - self._time_metrics['_last_step']
            time_elapsed = tn - self._time_metrics['_last_time']
            if steps_done>0:
                step_per_time = steps_done/time_elapsed
                local_dict['steps_per_time'] = step_per_time
                if 'max_steps' in local_dict:
                    steps_to_go = local_dict['max_steps'] - step_now
                    time_to_go = steps_to_go / step_per_time
                    local_dict['eta'] = time_to_go
            self._time_metrics['_last_time'] = tn
            self._time_metrics['_last_step'] = step_now

            rank_zero_info(self._format_table(local_dict))

    def set_description(self, *args) -> None:
        pass
        
    def set_postfix(self, local_dict: Dict) -> None:
        if 'loss' in local_dict:
            self.on_train_batch_end(local_dict)
        elif 'val_loss' in local_dict:
            self.on_validation(local_dict)

    def update(self) -> None:
        pass
        
    def _format_float(self, x: float):
        max_num_digits = self.col_width - 1
        if x>10**max_num_digits:
            fmt_str = '{:.' + str(max_num_digits-4) + 'g}'
        else:
            fmt_str = '{:.' + str(max_num_digits) + 'g}'
        return fmt_str.format(x)
    
    def _format_table(self, metrics: dict) -> str:
        # Formatting
        s = self.format_str
        fields = []
        for key in self.header:
            if key in metrics:
                if isinstance(metrics[key], float):
                    val = self._format_float(metrics[key])
                else:
                    val = metrics[key]
                fields.append(s.format(val))
            else:
                fields.append(s.format(''))
        line =  " | ".join(fields)
        return line

def upload_evaluations(results, name: str):
    outputs = torch.cat([tup[0]['stiffness'] for tup in results], dim=0).cpu().numpy()
    trues = torch.cat([tup[1]['stiffness'] for tup in results], dim=0).cpu().numpy()
    if 'train' in name:
        txt = ''
        ttl = 'Parity'
    elif 'val' in name:
        txt = 'val_'
        ttl = 'Val_parity'
    total_error = []
    for k in range(trues.shape[1]):
        data_table = np.column_stack((trues[:,k], outputs[:,k]))
        error = np.mean(np.abs(outputs[:,k] - trues[:,k]))/np.max(np.abs(trues[:,k]))
        total_error.append(error)
        table = wandb.Table(data=data_table, columns=[f'{txt}true{k}', f'{txt}pred{k}'])
        wandb.log({f"{ttl}{k}":wandb.plot.scatter(table=table, x=f"{txt}true{k}", y=f"{txt}pred{k}", title=f'#{k}, err={error*100:.2g}%')})
    max_error = max(total_error)
    avg_error = sum(total_error)/len(total_error)
    logging.info(f'{name}: average error={avg_error*100:.2g}%, max_error={max_error*100:.2g}%')
    wandb.log({f'{name}_avg_err':avg_error, f'{name}_max_err':max_error})


def plot_multi_parity(C_t, C_p, fig_kwargs: Optional[dict] = {}):
    fig = plt.figure(**fig_kwargs)
    for i in range(6):
        for j in range(i,6):
            ax = fig.add_subplot(6,6,6*i+j+1)
            # ax = axes[i,j]
            x = C_t[:,i,j]
            y = C_p[:,i,j]
            error = np.mean(np.abs(x-y)/np.abs(x).max())
            sns.histplot(x=x, y=y, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.annotate(f'{error*100:.1g}%', xy=(0.5,0.9), xycoords='axes fraction', ha='center')
    return fig

def local_plot_evaluations(results, name: str):
    y = torch.cat([r[0]['stiffness'] for r in results], dim=0)
    x = torch.cat([r[1]['stiffness'] for r in results], dim=0)
    rows, cols = np.triu_indices(6)
    C_t = np.zeros((y.shape[0], 6, 6))
    C_p = np.zeros((y.shape[0], 6, 6))
    for ii in range(21):
        i,j = rows[ii], cols[ii]
        C_t[:, i,j] = x[:, ii]
        C_p[:, i,j] = y[:, ii]
        
    ## %%
    fig = plot_multi_parity(C_t=C_t, C_p=C_p, fig_kwargs={'figsize':(10,10)})
    plt.savefig(name, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='w')


def log_matrix2(input: torch.Tensor, target: torch.Tensor, name: str):
    S = torch.zeros((6,13))

    imin = 0
    imax = 6
    for S_i in [input.clone(), target.clone()]:
        maxampl = torch.abs(S_i).max()
        S_i[torch.abs(S_i)<0.005*maxampl] = 0.0
        S[:, imin:imax] = S_i
        imin += 7
        imax += 7
    S[:,6] = torch.nan
    plt.imshow(S.numpy(), cmap='coolwarm')
    for i in range(6):
        for j in range(6):
            plt.text(j, i, f'{S[i,j]:.2g}', ha='center', va='center', fontsize=9)
        for j in range(7,13):
            plt.text(j, i, f'{S[i,j]:.2g}', ha='center', va='center', fontsize=9)

    error = torch.nn.functional.l1_loss(input, target)
    plt.text(6, 1.5, f'L1', ha='center', va='top')
    plt.text(6, 2, f'Error', ha='center', va='top')
    plt.text(6, 3, f'{error:.3f}', ha='center', va='bottom', fontsize=8)
    plt.xticks([])
    plt.yticks([])

    wandb.log({name: plt})

    plt.clf()
