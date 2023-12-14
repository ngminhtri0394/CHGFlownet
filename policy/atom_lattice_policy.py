from policy.graph_backbone import GraphPolicyBackbone
from torch.distributions import MultivariateNormal, VonMises, Normal,Categorical, Bernoulli
import torch
import torch.nn as nn

class GraphLatticeAtomPolicy(GraphPolicyBackbone):
    def __init__(self, dim=4, hidden_dim=64, n_hidden=2, output_dim=32, use_pretrain=True, 
                 pretrain_dim=128, n_element=8, device=torch.device('cuda'),
                 llocmin=4,llocmax=8,lsclmin=1,lsclmax=2,
                 alocmin=0.78,alocmax=2.36,aconmin=1.0,aconmax=2.0,
                 frlocmin=0.1,frlocmax=0.9,batchsize=32):
        super().__init__(dim, hidden_dim, n_hidden, output_dim, use_pretrain, pretrain_dim)
        self.prob_output = 15 + n_element
        self.n_element = n_element
        sg_dim = 64
        self.spacegroup_emb =nn.Embedding(230, sg_dim)
        self.outlayer = nn.Linear(output_dim + sg_dim, self.prob_output)
        self.device = device
        self.llocmin = llocmin
        self.llocmax = llocmax
        self.lsclmin = lsclmin
        self.lsclmax = lsclmax
        self.alocmin = alocmin
        self.alocmax = alocmax
        self.aconmin = aconmin
        self.aconmax = aconmax
        self.frlocmin = frlocmin
        self.frlocmax = frlocmax
        self.batchsize = batchsize

        self.PFs0 = nn.ParameterDict(
            {
                "len1_loc": nn.Parameter(torch.zeros(batchsize)),
                "len1_scl": nn.Parameter(torch.zeros(batchsize)),
                "len2_loc": nn.Parameter(torch.zeros(batchsize)),
                "len2_scl": nn.Parameter(torch.zeros(batchsize)),
                "len3_loc": nn.Parameter(torch.zeros(batchsize)),
                "len3_scl": nn.Parameter(torch.zeros(batchsize)),
                "a1_loc": nn.Parameter(torch.zeros(batchsize)),
                "a1_scl": nn.Parameter(torch.zeros(batchsize)),
                "a2_loc": nn.Parameter(torch.zeros(batchsize)),
                "a2_scl": nn.Parameter(torch.zeros(batchsize)),
                "a3_loc": nn.Parameter(torch.zeros(batchsize)),
                "a3_scl": nn.Parameter(torch.zeros(batchsize)),
                "frac_loc": nn.Parameter(torch.zeros(batchsize,3)),
                "type_logits": nn.Parameter(torch.zeros(batchsize,n_element)),
            })


    def forward(self, s, sub):
        graph, sg, lattice, fpretrain = s
        if graph != None:
            out = super().forward(s)
            sgemb = self.spacegroup_emb(sub.type(torch.int64))
            out = torch.cat((out, sgemb), dim=1)
            out = self.outlayer(out)

            len1_loc = self.llocmax*torch.sigmoid(out[...,0]) + self.llocmin
            len1_scl = self.lsclmax*torch.sigmoid(out[...,1]) + self.lsclmin
            len2_loc = self.llocmax*torch.sigmoid(out[...,2]) + self.llocmin
            len2_scl = self.lsclmax*torch.sigmoid(out[...,3]) + self.lsclmin
            len3_loc = self.llocmax*torch.sigmoid(out[...,4]) + self.llocmin
            len3_scl = self.lsclmax*torch.sigmoid(out[...,5]) + self.lsclmin

            a1_loc = self.alocmax*torch.sigmoid(out[...,6]) + self.alocmin
            a1_scl = self.aconmax*torch.sigmoid(out[...,7]) + self.aconmin
            a2_loc = self.alocmax*torch.sigmoid(out[...,8]) + self.alocmin
            a2_scl = self.aconmax*torch.sigmoid(out[...,9]) + self.aconmin
            a3_loc = self.alocmax*torch.sigmoid(out[...,10]) + self.alocmin
            a3_scl = self.aconmax*torch.sigmoid(out[...,11]) + self.aconmin

            frac_loc = self.frlocmax*torch.sigmoid(out[..., 12:15])+self.frlocmin
            type_logits = out[...,15:]
        else:
            len1_loc = self.llocmax*torch.sigmoid(self.PFs0['len1_loc']) + self.llocmin
            len1_scl = self.lsclmax*torch.sigmoid(self.PFs0['len1_loc']) + self.lsclmin
            len2_loc = self.llocmax*torch.sigmoid(self.PFs0['len1_loc']) + self.llocmin
            len2_scl = self.lsclmax*torch.sigmoid(self.PFs0['len1_loc']) + self.lsclmin
            len3_loc = self.llocmax*torch.sigmoid(self.PFs0['len1_loc']) + self.llocmin
            len3_scl = self.lsclmax*torch.sigmoid(self.PFs0['len1_loc']) + self.lsclmin

            a1_loc = self.alocmax*torch.sigmoid(self.PFs0['a1_loc']) + self.alocmin
            a1_scl = self.aconmax*torch.sigmoid(self.PFs0['a1_scl']) + self.aconmin
            a2_loc = self.alocmax*torch.sigmoid(self.PFs0['a2_loc']) + self.alocmin
            a2_scl = self.aconmax*torch.sigmoid(self.PFs0['a2_scl']) + self.aconmin
            a3_loc = self.alocmax*torch.sigmoid(self.PFs0['a3_loc']) + self.alocmin
            a3_scl = self.aconmax*torch.sigmoid(self.PFs0['a3_scl']) + self.aconmin

            frac_loc = self.frlocmax*torch.sigmoid(self.PFs0['frac_loc'])+self.frlocmin
            type_logits = self.PFs0['type_logits']

        return [len1_loc, len1_scl,
                    len2_loc, len2_scl,
                    len3_loc, len3_scl,
                    a1_loc, a1_scl,
                    a2_loc, a2_scl,
                    a3_loc, a3_scl,
                    frac_loc, type_logits]
    
    def to_dist(self, probs, logits):
        l1_dist = Normal(loc=logits[0], scale=logits[1])
        l2_dist = Normal(loc=logits[2], scale=logits[3])
        l3_dist = Normal(loc=logits[4], scale=logits[5])
        a1_dist = VonMises(loc=logits[6],concentration=logits[7])
        a2_dist = VonMises(loc=logits[8],concentration=logits[9])
        a3_dist = VonMises(loc=logits[10],concentration=logits[11])
        cov = torch.eye(3)*0.05
        frac_dist = MultivariateNormal(loc=logits[12],covariance_matrix=cov.to(device=self.device))
        atomtype_dist = Categorical(logits=logits[13])
        return [l1_dist, l2_dist, l3_dist,a1_dist, a2_dist,a3_dist,frac_dist, atomtype_dist]
        
        
class BWGraphLatticeAtomPolicy(GraphPolicyBackbone):
    def __init__(self, dim=4, hidden_dim=64, n_hidden=2, output_dim=32, use_pretrain=True, 
                 pretrain_dim=128, n_element=8, device=torch.device('cuda'),
                 llocmin=4,llocmax=8,lsclmin=1,lsclmax=2,
                 alocmin=0.78,alocmax=2.36,aconmin=1.0,aconmax=2.0,
                 frlocmin=0.1,frlocmax=0.9):
        super().__init__(dim, hidden_dim, n_hidden, output_dim, use_pretrain, pretrain_dim)
        self.n_element = n_element
        self.prob_output = 15 + n_element
        self.outlayer = nn.Linear(output_dim,self.prob_output)
        self.device = device
        self.llocmin = llocmin
        self.llocmax = llocmax
        self.lsclmin = lsclmin
        self.lsclmax = lsclmax
        self.alocmin = alocmin
        self.alocmax = alocmax
        self.aconmin = aconmin
        self.aconmax = aconmax
        self.frlocmin = frlocmin
        self.frlocmax = frlocmax
    
    def forward(self, s, subgoal):
        out = super().forward(s)
        out = self.outlayer(out)

        len1_loc = self.llocmax*torch.sigmoid(out[...,0]) + self.llocmin
        len1_scl = self.lsclmax*torch.sigmoid(out[...,1]) + self.lsclmin
        len2_loc = self.llocmax*torch.sigmoid(out[...,2]) + self.llocmin
        len2_scl = self.lsclmax*torch.sigmoid(out[...,3]) + self.lsclmin
        len3_loc = self.llocmax*torch.sigmoid(out[...,4]) + self.llocmin
        len3_scl = self.lsclmax*torch.sigmoid(out[...,5]) + self.lsclmin

        a1_loc = self.alocmax*torch.sigmoid(out[...,6]) + self.alocmin
        a1_scl = self.aconmax*torch.sigmoid(out[...,7]) + self.aconmin
        a2_loc = self.alocmax*torch.sigmoid(out[...,8]) + self.alocmin
        a2_scl = self.aconmax*torch.sigmoid(out[...,9]) + self.aconmin
        a3_loc = self.alocmax*torch.sigmoid(out[...,10]) + self.alocmin
        a3_scl = self.aconmax*torch.sigmoid(out[...,11]) + self.aconmin

        frac_loc = self.frlocmax*torch.sigmoid(out[..., 12:15])+self.frlocmin
        type_logits = out[...,15:]
        return [len1_loc, len1_scl,
                len2_loc, len2_scl,
                len3_loc, len3_scl,
                a1_loc, a1_scl,
                a2_loc, a2_scl,
                a3_loc, a3_scl,
                frac_loc, type_logits]
    
    def to_dist(self, probs, logits):
        l1_dist = Normal(loc=logits[0], scale=logits[1])
        l2_dist = Normal(loc=logits[2], scale=logits[3])
        l3_dist = Normal(loc=logits[4], scale=logits[5])
        a1_dist = VonMises(loc=logits[6],concentration=logits[7])
        a2_dist = VonMises(loc=logits[8],concentration=logits[9])
        a3_dist = VonMises(loc=logits[10],concentration=logits[11])
        cov = torch.eye(3)*0.05
        frac_dist = MultivariateNormal(loc=logits[12],covariance_matrix=cov.to(device=self.device))
        atomtype_dist = Categorical(logits=logits[13])
        return [l1_dist, l2_dist, l3_dist,a1_dist, a2_dist,a3_dist,frac_dist, atomtype_dist]