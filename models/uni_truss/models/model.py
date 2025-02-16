import torch
import numpy as np
from numpy import genfromtxt
import time
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from scipy import sparse
import torchvision
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from torch.autograd import Variable
from torch_scatter import scatter_mean

# from prefetch_generator import BackgroundGenerator
from .parameters import *

def dReLU(x):
    return (x-14) * ((x-14) > 0)

# class data_prefetcher():
#     def __init__(self, loader):
#         self.loader = iter(loader)
#         # self.stream = torch.cuda.Stream()
#         self.preload()
#
#     def preload(self):
#         try:
#             self.next_data = next(self.loader)
#         except StopIteration:
#             self.next_input = None
#             return
#
#     def next(self):
#         # torch.cuda.current_stream().wait_stream(self.stream)
#         data = self.next_data
#         self.preload()
#         return data

# class DataLoaderX(DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())

def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L    

class vaeModel(nn.Module):
    def __init__(self):
        super(vaeModel, self).__init__()
        ## encoder layers ##
        # adj_encoder
        self.en_adj_ch1, self.en_adj_ch2, self.en_adj_ch3, self.en_adj_ch4 = 12, 16, 16, 16
        self.en_adj_kernel1, self.en_adj_kernel2, self.en_adj_kernel3, self.en_adj_kernel4 = 3, 3, 2, 2
        self.en_adj_stride1, self.en_adj_stride2, self.en_adj_stride3, self.en_adj_stride4 = 1, 1, 1, 1
        self.en_adj_dim1, self.en_adj_dim2, self.en_adj_dim3, self.en_adj_dim4 = 512, 512, 512, 128

        self.en_x_ch1, self.en_x_ch2 = 20, 36
        self.en_x_kernel1, self.en_x_kernel2 = (3, 1), (2, 1)
        self.en_x_stride1, self.en_x_stride2 = (2, 1), 1
        self.en_x_dim1, self.en_x_dim2 = 12, 1
        self.en_x_hd1, self.en_x_hd2, self.en_x_hd3, self.en_x_hd4 = 256, 256, 128, latent_dim
       
        self.en_adj_fc1 = nn.Linear(adj_vec_dim, self.en_adj_dim1)
        self.en_adj_fc2 = nn.Linear(self.en_adj_dim1, self.en_adj_dim2)
        self.en_adj_fc3 = nn.Linear(self.en_adj_dim2, self.en_adj_dim3)
        self.en_adj_fc4 = nn.Linear(self.en_adj_dim3, self.en_adj_dim4)
        self.en_adj_fc5 = nn.Linear(self.en_adj_dim4, a_dim + ax_dim)

        self.adj_fc_mu = nn.Linear(a_dim + ax_dim,  a_dim + ax_dim)
        self.adj_fc_std = nn.Linear(a_dim + ax_dim, a_dim + ax_dim)

        # x_encoder
        x_hidden_dim = [dim*numNodes, 640, 640, 640, 512, ax_dim + x_dim]
        self.en_x_fc1 = nn.Linear(x_hidden_dim[0], x_hidden_dim[1])
        self.en_x_fc2 = nn.Linear(x_hidden_dim[1], x_hidden_dim[2])
        self.en_x_fc3 = nn.Linear(x_hidden_dim[2], x_hidden_dim[3])
        self.en_x_fc4 = nn.Linear(x_hidden_dim[3], x_hidden_dim[4])
        self.en_x_fc5 = nn.Linear(x_hidden_dim[4], x_hidden_dim[5])
   
        self.x_fc_mu = nn.Linear(ax_dim + x_dim,  ax_dim + x_dim)
        self.x_fc_std = nn.Linear(ax_dim + x_dim, ax_dim + x_dim)

        # adj_decoder
        self.adj_de_fc = nn.Linear(24, self.en_x_hd3)
        self.x_de_fc = nn.Linear(24, self.en_x_hd3)

        self.de_adj_fc1 = nn.Linear(a_dim + ax_dim, self.en_adj_dim4)
        self.de_adj_fc2 = nn.Linear(self.en_adj_dim4, self.en_adj_dim3)
        self.de_adj_fc3 = nn.Linear(self.en_adj_dim3, self.en_adj_dim2)
        self.de_adj_fc4 = nn.Linear(self.en_adj_dim2, self.en_adj_dim1)
        self.de_adj_fc5 = nn.Linear(self.en_adj_dim1, adj_vec_dim)
       
        # x_decoder
        de_x_hidden_dim = [x_dim + ax_dim, 640, 640, 512, 256, dim*numNodes]
        self.de_x_fc1 = nn.Linear(de_x_hidden_dim[0], de_x_hidden_dim[1])
        self.de_x_fc2 = nn.Linear(de_x_hidden_dim[1], de_x_hidden_dim[2])
        self.de_x_fc3 = nn.Linear(de_x_hidden_dim[2], de_x_hidden_dim[3])
        self.de_x_fc4 = nn.Linear(de_x_hidden_dim[3], de_x_hidden_dim[4])
        self.de_x_fc5 = nn.Linear(de_x_hidden_dim[4], de_x_hidden_dim[5])

        self.activation = nn.ReLU()

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(logvar)
        eps = torch.FloatTensor(logvar.size()[0],1).normal_(0,1)
        eps = eps.to(mu.device)
        eps  = eps.expand(sigma.size())
        return mu + sigma*eps

    def encoder(self, adj, x):
        
        adj = self.activation(self.en_adj_fc1(adj))
        adj = self.activation(self.en_adj_fc2(adj))
        adj = self.activation(self.en_adj_fc3(adj))
        adj = self.activation(self.en_adj_fc4(adj))
        adj = self.en_adj_fc5(adj)


        x = self.activation(self.en_x_fc1(x))
        x = self.activation(self.en_x_fc2(x))
        x = self.activation(self.en_x_fc3(x))
        x = self.activation(self.en_x_fc4(x))
        x = self.en_x_fc5(x)

        adj_mu = self.adj_fc_mu(adj)
        adj_log_var = self.adj_fc_std(adj)

        adj_z = self.reparameterize(adj_mu, adj_log_var)

        x_mu = self.x_fc_mu(x)
        x_log_var = self.x_fc_std(x)

        x_z = self.reparameterize(x_mu, x_log_var)

        mu_tot = torch.cat((adj_mu, x_mu), dim = -1)
        log_var_tot = torch.cat((adj_log_var, x_log_var), dim = -1)

        mu = torch.zeros([len(mu_tot), a_dim + ax_dim + x_dim]).to(x.device)
        log_var = torch.zeros([len(mu_tot), a_dim + ax_dim + x_dim]).to(x.device)

        mu[:,:a_dim] = mu_tot[:,:a_dim]
        mu[:,a_dim:(a_dim + ax_dim)] = 0.5*(mu_tot[:,a_dim:a_dim + ax_dim] + mu_tot[:,a_dim + ax_dim:a_dim+2*ax_dim])
        mu[:,a_dim+ax_dim:] = mu_tot[:,a_dim+2*ax_dim:]

        log_var[:,:a_dim] = log_var_tot[:,:a_dim]
        log_var[:,a_dim:(a_dim + ax_dim)] = 0.5*(log_var_tot[:,a_dim:a_dim + ax_dim] + log_var_tot[:,a_dim + ax_dim:a_dim+2*ax_dim])
        log_var[:,a_dim+ax_dim:] = log_var_tot[:,a_dim+2*ax_dim:]

        z = self.reparameterize(mu, log_var)

        return z, mu, log_var

    def decoder(self, z):

        adj_de_input = z[:,:a_dim+ax_dim]
        x_de_input = z[:,a_dim:]

        adj = self.activation(self.de_adj_fc1(adj_de_input))
        adj = self.activation(self.de_adj_fc2(adj))
        adj = self.activation(self.de_adj_fc3(adj))
        adj = self.activation(self.de_adj_fc4(adj))
        adj = self.de_adj_fc5(adj)
        
        x = self.activation(self.de_x_fc1(x_de_input))
        x = self.activation(self.de_x_fc2(x))
        x = self.activation(self.de_x_fc3(x))
        x = self.activation(self.de_x_fc4(x))
        x = self.de_x_fc5(x)
        
        adj_out = torch.sigmoid(adj)

        x_out = torch.sigmoid(x) - 0.5
        return adj_out, x_out

class cModel(nn.Module):
    def __init__(self):
        super(cModel, self).__init__()
        self.dropout_p = 0.

        self.model = torch.nn.Sequential()
        self.model.add_module('e_fc1', nn.Linear(a_dim + ax_dim + x_dim, c_hidden_dim[0], bias = False))
        self.model.add_module('e_relu1',nn.ReLU())
        for i in range(1, len(c_hidden_dim)):
            self.model.add_module('e_fc' + str(i+1),nn.Linear(c_hidden_dim[i - 1], c_hidden_dim[i], bias = False))
            self.model.add_module('e_relu' + str(i+1),nn.ReLU())
            self.model.add_module('e_dropout' + str(i+1), nn.Dropout(self.dropout_p))
        self.model.add_module('de_out',nn.Linear(c_hidden_dim[-1], paramDim, bias = False))

    def forward(self, z):
        # z = scatter_mean(z, batch, dim=0)
        out = self.model(z)
        return out

def kld_loss(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            # m.bias.data.fill_(0.01)



