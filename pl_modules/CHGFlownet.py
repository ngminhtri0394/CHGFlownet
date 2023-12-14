import argparse
import gzip
import os
import pickle
import warnings
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig

warnings.filterwarnings('ignore')
from sampling.action_sampling import HierarchicalActionSampler, BackwardActionsSampler
from sampling.trajectory_sampling import TrajectoriesSampler
from env.crystalenv import HierCrystalEnv
from pl_modules.proxy import M3gnetDGL_Proxy
import torch
# torch.set_printoptions(profile="full")
from policy.spacegroup_policy import GraphSpaceGroupPolicy, BWGraphSpaceGroupPolicy
from policy.atom_lattice_policy import GraphLatticeAtomPolicy, BWGraphLatticeAtomPolicy
from tqdm import tqdm
import os
from tqdm import tqdm
from functools import partialmethod
from tqdm import tqdm, trange
from metrics.eval_metrics import *
from pl_modules.reward import reward_functions_dict

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CHGFlownet():
    def __init__(self,
                 max_blocks,
                 device,
                 lr,
                 lr_Z,
                 scheduler_milestone,
                 gamma_scheduler,
                 initlogZ,
                 n_iterations,
                 alpha_schedule,
                 alpha,
                 clampmin,
                 clampmax,
                 batch_size,
                 save_freq,
                 phidden_dim,
                 use_pretrain,
                 pretrain_model_name,
                 proxy_model,
                 with_stop,
                 min_stop,
                 policy_nn,
                 ele_set,
                 frlocmin,
                 frlocmax,
                 llocmin,
                 llocmax,
                 lsclmin,
                 lsclmax,
                 frcovmin,
                 frcovmax,
                 alocmin,
                 alocmax,
                 aconmin,
                 aconmax,
                 max_ele,
                 req_ele,
                 max_atom,
                 vpen_min,
                 vpen_max,
                 vpen_minmax,
                 reward):
        self.reward = reward_functions_dict[reward]
        self.vpen_min = vpen_min
        self.vpen_max = vpen_max
        self.vpen_minmax = vpen_minmax
        self.max_blocks = max_blocks
        self.phidden_dim = phidden_dim
        self.with_stop = with_stop
        self.min_stop = min_stop
        self.device = torch.device(device)
        self.lr = lr
        self.lr_Z = lr_Z
        self.scheduler_milestone = scheduler_milestone
        self.gamma_scheduler = gamma_scheduler
        self.initlogZ = initlogZ
        self.n_iterations = n_iterations
        self.alpha_schedule = alpha_schedule
        self.alpha = alpha
        self.clampmin = clampmin
        self.clampmax = clampmax
        self.batch_size = batch_size
        self.sampled_mols = []
        self.sampled_reward = []
        self.sampled_bs = []
        self.train_infos = []
        self.save_freq = save_freq
        self.use_pretrain = use_pretrain
        self.pretrain_model_name = pretrain_model_name
        self.logZ = nn.Parameter(torch.tensor(self.initlogZ).to(device=self.device))   
        self.proxy_model = proxy_model
        self.proxy = M3gnetDGL_Proxy(self.proxy_model)
        self.policy_nn = policy_nn
        self.ele_set = ele_set
        self.frlocmin = frlocmin
        self.frlocmax = frlocmax
        self.llocmin = llocmin
        self.llocmax = llocmax
        self.lsclmin = lsclmin
        self.lsclmax = lsclmax
        self.frcovmin = frcovmin
        self.frcovmax = frcovmax
        self.alocmin = alocmin
        self.alocmax = alocmax
        self.aconmin = aconmin
        self.aconmax = aconmax
        if self.pretrain_model_name == 'matformer':
            self.pretrain_dim = 128
        else: 
            self.pretrain_dim = 256
        if self.policy_nn == 'graph':
            self.env = HierCrystalEnv(device=self.device, pretrain_model=self.pretrain_model_name, ele_set=self.ele_set,batch_size=batch_size)
        self.num_elementset = len(self.env.atoms)
        req_elementidx = [self.env.atoms.index(e) for e in req_ele]
        self.req_config = {'max_ele':max_ele,
                           'req_ele':req_elementidx,
                           'max_atom':max_atom,
                           'len_ele_list': len(self.env.atoms),
                           'ele_choice': self.env.atoms}
        self.env.set_req(self.req_config)
        self.env.reset()
        self.init_policy()
        self.init_sampler()

    def init_policy(self, path=None):
        if path == None:
                self.sgpolicy = GraphSpaceGroupPolicy(use_pretrain=self.use_pretrain,
                                            hidden_dim=self.phidden_dim,
                                            pretrain_dim=self.pretrain_dim).to(device=self.device)
                self.latticeatompolicy = GraphLatticeAtomPolicy(use_pretrain=self.use_pretrain,
                                            hidden_dim=self.phidden_dim,
                                            pretrain_dim=self.pretrain_dim,
                                            aconmax=self.aconmax,
                                            aconmin=self.aconmin,
                                            alocmax=self.alocmax,
                                            alocmin=self.alocmin,
                                            llocmax=self.llocmax,
                                            llocmin=self.llocmin,
                                            lsclmax=self.lsclmax,
                                            lsclmin=self.lsclmin,
                                            frlocmax=self.frlocmax,
                                            frlocmin=self.frcovmin,
                                            n_element=self.num_elementset).to(device=self.device)
                self.bwsgpolicy = BWGraphSpaceGroupPolicy(use_pretrain=self.use_pretrain,
                                                    hidden_dim=self.phidden_dim,
                                                    pretrain_dim=self.pretrain_dim).to(device=self.device)
                self.bwlatticeatompolicy = BWGraphLatticeAtomPolicy(use_pretrain=self.use_pretrain,
                                            hidden_dim=self.phidden_dim,
                                            pretrain_dim=self.pretrain_dim,
                                            aconmax=self.aconmax,
                                            aconmin=self.aconmin,
                                            alocmax=self.alocmax,
                                            alocmin=self.alocmin,
                                            llocmax=self.llocmax,
                                            llocmin=self.llocmin,
                                            lsclmax=self.lsclmax,
                                            lsclmin=self.lsclmin,
                                            frlocmax=self.frlocmax,
                                            frlocmin=self.frcovmin,
                                            n_element=self.num_elementset).to(device=self.device)
        else:
            self.sgpolicy = torch.load(f'{path}/'+'policy_fp_sg.pt',map_location=torch.device('cpu')).to(device=self.device)
            self.latticepolicy = torch.load(f'{path}/'+'policy_fp_lt.pt',map_location=torch.device('cpu')).to(device=self.device)
            self.atompolicy = torch.load(f'{path}/'+'policy_fp_at.pt',map_location=torch.device('cpu')).to(device=self.device)
            self.bwsgpolicy = torch.load(f'{path}/'+'policy_bw_sg.pt',map_location=torch.device('cpu')).to(device=self.device)
            self.bwlatticepolicy = torch.load(f'{path}/'+'policy_bw_lt.pt',map_location=torch.device('cpu')).to(device=self.device)
            self.bwatompolicy = torch.load(f'{path}/'+'policy_bw_at.pt',map_location=torch.device('cpu')).to(device=self.device)
        
        self.bwhpolicylist = [self.bwsgpolicy, self.bwlatticeatompolicy]
        self.hpolicylist = [self.sgpolicy, self.latticeatompolicy]

                

    def init_sampler(self):
        self.action_sampler = HierarchicalActionSampler(estimators=self.hpolicylist,
                                                                min_stop=self.min_stop,
                                                                req_config=self.req_config)
        self.bw_sampler = BackwardActionsSampler(estimators=self.bwhpolicylist,
                                                                    req_config=self.req_config)
        self.trajectory_sampling = TrajectoriesSampler(action_sampler=self.action_sampler, 
                                                        bwaction_sampler= self.bw_sampler, 
                                                        env=self.env,
                                                        max_blocks=self.max_blocks,
                                                        min_stop=self.min_stop,
                                                        req_config=self.req_config
                                                        )


    
    def save_info(self,iter):
        exp_dir = HydraConfig.get().run.dir
        sampled = zip(self.sampled_mols,self.sampled_reward, self.sampled_bs)
        if not os.path.isdir(f'{exp_dir}/'+'saved_data/'):
            os.makedirs(f'{exp_dir}/'+'saved_data/')
        
        pickle.dump(sampled,
                        gzip.open(f'{exp_dir}/' +'saved_data/' + str(iter) + '_sampled_mols.pkl.gz', 'wb'))

        pickle.dump(self.train_infos,
                        gzip.open(f'{exp_dir}/' +'saved_data/' + str(iter) + '_train_info.pkl.gz', 'wb'))
        
        torch.save(self.hpolicylist[0], f'{exp_dir}/'+'policy_fp_sg.pt')
        torch.save(self.hpolicylist[1], f'{exp_dir}/'+'policy_fp_alt.pt')

        torch.save(self.bwhpolicylist[0], f'{exp_dir}/'+'policy_bw_sg.pt')
        torch.save(self.bwhpolicylist[1], f'{exp_dir}/'+'policy_bw_alt.pt')

        torch.save(self.logZ, f'{exp_dir}/'+'logz.pt')
        
    def train_model_with_proxy(self):
        optimizer = torch.optim.Adam(self.hpolicylist[0].parameters(), lr=self.lr)
        optimizer.add_param_group({"params": self.hpolicylist[1].parameters(), "lr": self.lr})

        optimizer.add_param_group({"params": self.bwhpolicylist[0].parameters(), "lr": self.lr})
        optimizer.add_param_group({"params": self.bwhpolicylist[1].parameters(), "lr": self.lr})
        optimizer.add_param_group({"params": [self.logZ], "lr": self.lr_Z})
        # self.scheduler_milestone = 5000
        # self.gamma_scheduler = 1.0
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[i * self.scheduler_milestone for i in range(1, 10)],
            gamma=self.gamma_scheduler,
        )

        ####################
        current_alpha = self.alpha * self.alpha_schedule
        tr = trange(self.n_iterations)
        currentmax = 0
        for i in tr:
            print('Epoch: ', i)
            if i % 1000 == 0:
                current_alpha = max(current_alpha / self.alpha_schedule, 1.0)
                print(f"current optimizer LR: {optimizer.param_groups[0]['lr']}")

            optimizer.zero_grad()

            trajectories, actionss, logprobs, all_logprobs,states_reps = self.trajectory_sampling.sample_trajectories()
            last_states = trajectories[:,-1]
            
            reward,bs = self.reward(last_states, self.proxy, self.vpen_min, self.vpen_max, self.vpen_minmax)
            
            self.sampled_mols.append(last_states)
            self.sampled_reward.append(reward)
            self.sampled_bs.append(bs)
            logrewards = torch.Tensor(reward).to(device=self.device).log()
            max_reward = np.max(reward)
            mean_bs = np.mean(bs)

            bw_logprobs = self.trajectory_sampling.evaluate_backward_logprobs(trajectories, actionss, self.batch_size, states_reps)

            loss = torch.mean((self.logZ + logprobs - bw_logprobs - logrewards) ** 2)
            loss.backward()
            if torch.isnan(loss):
                print(self.logZ)
                print(logprobs)
                print(bw_logprobs)
                print(logrewards)
                print(reward)
                raise ValueError('loss is nan')
            print('Loss: ', loss.item())
            print('Max reward: ', max_reward)
            mean_reward = np.mean(reward)
            print('Mean reward: ', np.mean(reward))
            if max_reward > currentmax:
                currentmax = max_reward
            print('Current max: ', currentmax)
            self.train_infos.append((loss.item(),max_reward,mean_reward,currentmax,mean_bs))
            # clip the gradients for bw_model
            for model_bw in self.bwhpolicylist:
                for p in model_bw.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(self.clampmin, self.clampmax).nan_to_num_(0.0)
            for model in self.hpolicylist:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(self.clampmin, self.clampmax).nan_to_num_(0.0)
            optimizer.step()
            scheduler.step()
            if i % self.save_freq == 0:
                self.save_info(i)
            tr.set_postfix(Loss=loss.item(),MeanReward=max_reward,logz=self.logZ.item())
            self.env.reset()

    def load_sampling_model(self,save_path):
        self.init_policy(save_path)
        self.init_sampler()