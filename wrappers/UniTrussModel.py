import datetime
import os
from argparse import Namespace

import wandb
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import DataLoader
from torch_scatter import scatter_add, scatter_mean
from tqdm import tqdm

from evaluation.prediction_eval import calculate_metrics
from models.uni_truss.models.model import vaeModel, cModel, kld_loss
from models.uni_truss.models.utils import *
from models.uni_truss.models.parameters import *
from wrappers.BaseModel import BaseModel
import torch
from torch.nn import functional as F
import time


class UniTrussModel(BaseModel):
    def __init__(self, model_name, dataset_name, device=torch.device('cuda'), root_path='../'):
        config = self.load_config(os.path.join(root_path, 'configs', model_name, 'config.yml'))
        super().__init__(model_name, dataset_name, device, root_path, config)
        self.ckpt = None
        self.model = None
        self.c_model = None

    def load_model(self, checkpoint_path=None):

        self.model = vaeModel()
        self.c_model = cModel()
        if checkpoint_path is not None:
            print('Load checkpoint from', checkpoint_path)
            self.c_model.load_state_dict(torch.load(checkpoint_path+'/best_c_model.pt', map_location=self.device))
            self.model.load_state_dict(torch.load(checkpoint_path+'/best_model.pt', map_location=self.device))

        return self.model, self.c_model

    def train(self):
        self.model.train()
        self.c_model.train()
        self.model = self.model.to(self.device)
        self.c_model = self.c_model.to(self.device)

        # train_loader = DataLoader(
        #     dataset=self.train_data,
        #     batch_size=self.config['training']['batch_size'],
        #     shuffle=True,
        # )
        val_loader = DataLoader(
            dataset=self.val_data,
            batch_size=self.config['training']['valid_batch_size'],
            shuffle=False,
        )

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

        optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.c_model.parameters()), lr=learningRate)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

        best_metrics = {
            'epoch': 0,
            'loss': float('inf'),
            'val_loss': float('inf'),
        }
        val_loss = float('inf')
        loss = float('inf')
        batch_size = self.config['training']['batch_size']
        bad_count = 0
        for epoch in range(self.config['training']['epochs']):
            mean_time_per_batch = 0
            step = 0
            while step < len(self.train_data):
                xs = []
                adjs = []
                cs = []
                for batch_idx in range(batch_size):
                    if not step < len(self.train_data):
                        break
                    data = self.train_data[step]
                    step += 1
                    x = data.frac_coords
                    adj = torch.zeros(numNodes, numNodes)
                    row, col = data.edge_index
                    adj[row, col] = 1.
                    adj = torch.triu(adj)
                    row, col = torch.triu_indices(numNodes, numNodes, offset=0)
                    adj = adj[row, col].view(1, -1)
                    x = F.pad(x, (0, 0, 0, numNodes - len(x))).view(1, -1)

                    if self.config['pred_property'] == 'young':
                        true_y = data.y[:, :3]
                    elif self.config['pred_property'] == 'shear':
                        true_y = data.y[:, 3:6]
                    elif self.config['pred_property'] == 'poisson':
                        true_y = data.y[:, 6:]
                    else:
                        true_y = data.y
                    c = true_y
                    xs.append(x)
                    adjs.append(adj)
                    cs.append(c)
                adj = torch.cat(adjs)
                c = torch.cat(cs)
                x = torch.cat(xs)

                start_time = time.time()

                adj, x, c = adj.to(self.device), x.to(self.device), c.to(self.device)
                optimizer.zero_grad()
                encoded, mu, std = self.model.encoder(adj, x)
                c_input = encoded if self.config["model"]["add_noise"] else mu
                c_pred = self.c_model(c_input)
                adj_decoded, x_decoded = self.model.decoder(encoded)

                adj_train_mse = recon_criterion(adj_decoded, adj)
                x_train_mse = recon_criterion(x_decoded, x)
                train_kld = kld_loss(mu, std)

                c_weight = self.config["loss_weights"]["c_weight"][epoch % len(self.config["loss_weights"]["c_weight"])]
                # c_train_mse = stiffness_weighted_loss(c_pred, c)
                c_train_mse = recon_criterion(c_pred, c)

                loss = (
                        adj_train_mse +
                        x_train_mse * self.config["loss_weights"]["x_weight"] +
                        train_kld +
                        c_train_mse * c_weight
                )
                loss.backward()
                optimizer.step()
                end_time = time.time()
                mean_time_per_batch += end_time - start_time
                # print(f"Epoch {epoch}, step {step}: Train loss: {loss}, closs={c_train_mse}, kld={train_kld}, xloss={x_train_mse} Val loss: {val_loss}")
            print(f"Epoch {epoch}, step {step}: Train loss: {loss}, closs={c_train_mse}, kld={train_kld}, xloss={x_train_mse} Val loss: {val_loss}, Mean time: {mean_time_per_batch / step}")
            if use_wandb == True:
                wandb.log({"Loss/train": loss})


            if (epoch % self.config['training']['val_check_interval'] == 0.) or (epoch == self.config['training']['epochs']-1):
                val_loss = self.evaluate(val_loader)
                if use_wandb == True:
                    wandb.log({"Loss/val": val_loss})

                if val_loss < best_metrics['val_loss']:
                    print('Saving best model to', self.config['output_dir'])
                    best_metrics.update({
                        'epoch': epoch,
                        'loss': loss,
                        'val_loss': val_loss,
                    })
                    os.makedirs(self.config['output_dir'], exist_ok=True)
                    torch.save(self.c_model.state_dict(), self.config['output_dir'] + '/best_c_model.pt')
                    torch.save(self.model.state_dict(), self.config['output_dir'] + '/best_model.pt')
                else:
                    bad_count += 1

            scheduler.step()
            if bad_count >= self.config['training']['patience']:
                val_loss = self.evaluate(val_loader)
                if use_wandb == True:
                    wandb.log({"Loss/val": val_loss})

                if val_loss < best_metrics['val_loss']:
                    print('Saving best model to', self.config['output_dir'])
                    best_metrics.update({
                        'epoch': epoch,
                        'loss': loss,
                        'val_loss': val_loss,
                    })
                    os.makedirs(self.config['output_dir'], exist_ok=True)
                    torch.save(self.c_model.state_dict(), self.config['output_dir'] + '/best_c_model.pt')
                    torch.save(self.model.state_dict(), self.config['output_dir'] + '/best_model.pt')
                break


    def evaluate(self, loader):
        print('Evaluating')
        self.model.eval()
        self.c_model.eval()
        adj_test_mse, x_test_mse, c_test_mse, test_kld_loss = 0., 0., 0., 0.

        with torch.no_grad():
            for data in tqdm(loader):
                x = data.frac_coords
                adj = torch.zeros(numNodes, numNodes)
                row, col = data.edge_index
                adj[row, col] = 1.
                adj = torch.triu(adj)
                row, col = torch.triu_indices(numNodes, numNodes, offset=0)
                adj = adj[row, col].view(1, -1)
                x = F.pad(x, (0, 0, 0, numNodes - len(x))).view(1, -1)
                if self.config['pred_property'] == 'young':
                    true_y = data.y[:, :3]
                elif self.config['pred_property'] == 'shear':
                    true_y = data.y[:, 3:6]
                elif self.config['pred_property'] == 'poisson':
                    true_y = data.y[:, 6:]
                else:
                    true_y = data.y
                c = true_y

                adj, x, c = adj.to(self.device), x.to(self.device), c.to(self.device)
                encoded, mu, std = self.model.encoder(adj, x)
                c_input = encoded if self.config["model"]["add_noise"] else mu
                c_pred = self.c_model(c_input)
                adj_decoded, x_decoded = self.model.decoder(encoded)

                adj_test_mse += recon_criterion(adj_decoded, adj).item()
                x_test_mse += recon_criterion(x_decoded, x).item()
                test_kld_loss += kld_loss(mu, std).item()
                c_test_mse += recon_criterion(c_pred, c).item()

        return sum([adj_test_mse, x_test_mse, c_test_mse, test_kld_loss])


    def test(self):
        print("Teste and generate...")
        os.makedirs(self.config['gen_path'], exist_ok=True)
        # print('Load checkpoint from', self.config['output_dir'])
        # self.c_model.load_state_dict(torch.load(self.config['output_dir'] + '/best_c_model.pt', map_location=self.device))
        # self.model.load_state_dict(torch.load(self.config['output_dir'] + '/best_model.pt', map_location=self.device))
        self.model.eval()
        self.c_model.eval()
        self.c_model = self.c_model.to(self.device)
        self.model = self.model.to(self.device)
        test_loader = DataLoader(
            dataset=self.test_data,
            batch_size=self.config['training']['valid_batch_size'],
            shuffle=False,
        )

        loader_data = []
        gen_data = []

        frac_coords = []
        lengths = []
        angles = []
        num_atoms = []
        edge_indexs = []
        atom_types = []

        pred_all = []
        targe_all = []
        mean_time_per_batch = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader)):
                batch_dict = {
                    key: getattr(data, key) for key in data.keys()
                }
                loader_data.append(batch_dict)

                x = data.frac_coords
                adj = torch.zeros(numNodes, numNodes)
                row, col = data.edge_index
                adj[row, col] = 1.
                adj = torch.triu(adj)
                row, col = torch.triu_indices(numNodes, numNodes, offset=0)
                adj = adj[row, col].view(1, -1)
                x = F.pad(x, (0, 0, 0, numNodes - len(x))).view(1, -1)
                if self.config['pred_property'] == 'young':
                    true_y = data.y[:, :3]
                elif self.config['pred_property'] == 'shear':
                    true_y = data.y[:, 3:6]
                elif self.config['pred_property'] == 'poisson':
                    true_y = data.y[:, 6:]
                else:
                    true_y = data.y
                c = true_y

                adj, x, c = adj.to(self.device), x.to(self.device), c.to(self.device)

                start_time = time.time()
                encoded, mu, std = self.model.encoder(adj, x)
                c_input = mu
                c_pred = self.c_model(c_input)
                # adj_decoded, x_decoded = self.model.decoder(encoded)
                # adj_decoded = adj_decoded.view(numNodes, numNodes)
                # edge_index = torch.nonzero(adj_decoded)

                # x_decoded=x_decoded.reshape(-1,3)[:data.num_atoms]
                # frac_coords.append(x_decoded)
                # lengths.append(torch.tensor([1,1,1]))
                # angles.append(torch.tensor([0,0,0]))
                # edge_indexs.append(edge_index)
                # atom_types.append(0)
                # num_atoms.append(data.num_atoms)


                # lattice_name = os.path.join(self.config['gen_path'], str(i))
                # np.savez(lattice_name,
                #          atom_types=0,
                #          lengths=np.array([1.,1.,1.]),
                #          angles=np.array([90,90,90]),
                #          frac_coords=x.numpy(),
                #          edge_index=edge_index,
                #          prop_list=c_pred.numpy()
                #          )
                end_time=time.time()
                mean_time_per_batch += end_time-start_time
                pred_all.append(c_pred)
                targe_all.append(true_y)

        # torch.save(loader_data, self.config['gen_path'] + '/loader_data.pt')

        pred_all = torch.cat(pred_all, dim=0).cpu().numpy()
        targe_all = torch.cat(targe_all, dim=0).cpu().numpy()
        r2, nrmse, mae = calculate_metrics(pred_all, targe_all)
        if self.config['wandb_args']['use_wandb'] and wandb.run:
            wandb.log({"Test/R2": r2, 'NRMSE': nrmse, 'MAE': mae})
        print(f'R2:{r2}, NRMSE:{nrmse}, MAE:{mae}, Mean_time_per_batch:{mean_time_per_batch / len(test_loader)}')


        # frac_coords  = torch.cat(frac_coords)
        # lengths = torch.cat(lengths)
        # angles = torch.cat(angles)
        # num_atoms= torch.tensor(num_atoms)
        # # edge_indexs = torch.stack(edge_indexs)
        # atom_types = torch.tensor(atom_types)
        # torch.save({
        #     'frac_coords': frac_coords.cpu(),
        #     'num_atoms': num_atoms.cpu(),
        #     'atom_types': atom_types,
        #     'lengths': lengths.cpu(),
        #     'edge_index': edge_indexs,
        #     'angles': angles.cpu(),
        # }, self.config['gen_path'] + '/recon_data.pt')
        return r2, nrmse, mae

    def save_results(self, path='results/'):
        pass

