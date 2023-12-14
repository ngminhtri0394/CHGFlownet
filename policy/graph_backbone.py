import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import global_mean_pool as gmp
from torch_geometric.nn import GCNConv

class ResidualBlock(torch.nn.Module):
    def __init__(self, outfeature):
        super(ResidualBlock, self).__init__()
        self.outfeature = outfeature
        self.gcn = GCNConv(outfeature,outfeature)
        self.ln = torch.nn.Linear(outfeature, outfeature, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        identity = x
        out = self.gcn(x, edge_index)
        out = self.relu(out)
        out = self.ln(out)
        out += identity
        out = self.relu(out)
        return out

class GraphPolicyBackbone(nn.Module):
    def __init__(self, dim=4, hidden_dim=64, n_hidden=2, output_dim=32,use_pretrain=True,pretrain_dim=128):
        super().__init__()
        self.use_pretrain = use_pretrain
        self.relu = nn.ReLU()
        self.n_hidden = n_hidden
        self.conv1 = GCNConv(dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim*2)
        self.rblock_xd = ResidualBlock(hidden_dim*2)
        self.fc = torch.nn.Linear(hidden_dim*2, hidden_dim)

        self.emb_lattice = nn.Sequential(
                nn.Linear(10, hidden_dim),
                nn.ELU(),
                *[
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ELU(),
                    )
                    for _ in range(n_hidden)
                ],
            )
        
        self.emb_pretrain = nn.Sequential(
                nn.Linear(pretrain_dim, hidden_dim),
                nn.ELU(),
                *[
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ELU(),
                    )
                    for _ in range(n_hidden)
                ],
            )
        
        self.emb_sg = nn.Embedding(230, output_dim)
        if use_pretrain:
            self.output_layer = nn.Linear(hidden_dim*4, output_dim)
        else:
            self.output_layer = nn.Linear(hidden_dim*3, output_dim)

    
    def forward(self, s):
        graph, sg, lattice, fpretrain = s
        x_graph, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x_graph = torch.squeeze(x_graph)
        x_graph = self.conv1(x_graph, edge_index)
        x_graph = self.relu(x_graph)
        x_graph = self.conv2(x_graph, edge_index)
        x_graph = self.relu(x_graph)
        for i in range(self.n_hidden):
            x_graph = self.rblock_xd(x_graph, edge_index)
        x_graph = gmp(x_graph, batch)
        x_graph = self.relu(self.fc(x_graph))

        x_lattice = self.emb_lattice(lattice)
        x_sg = self.emb_sg(sg)

        if self.use_pretrain:
            x_pretrain = self.emb_pretrain(fpretrain)
            out = torch.cat((x_graph,x_lattice,x_pretrain,x_sg), dim=1)
        else:
            out = torch.cat((x_graph,x_lattice, x_sg), dim=1)
        out = self.output_layer(out)
        return out

