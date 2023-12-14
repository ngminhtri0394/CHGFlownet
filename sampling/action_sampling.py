from abc import ABC, abstractmethod
from policy.graph_backbone import GraphPolicyBackbone
from typing import List
from pl_modules.structure import CrystalStructureCData
import torch
import math
import numpy as np

class ActionsSampler(ABC):
    def __init__(self,
                 estimators: List[GraphPolicyBackbone],
                 req_config,
                temperature: float = 1.0,
                sf_bias: float = 0.0,
                epsilon: float = 0.0) -> None:
        super().__init__()
        self.req_config = req_config
        self.estimators = estimators
        self.temperature = temperature
        self.sf_bias = sf_bias
        self.epsilon = epsilon
        

    @abstractmethod
    def sample(self, states):
        """
        Args:
            states (States): A batch of states.

        Returns:
            Tuple[Tensor[batch_size], Tensor[batch_size]]: A tuple of tensors containing the log probabilities of the sampled actions, and the sampled actions.
        """
        pass


class HierarchicalActionSampler(ActionsSampler):
    """
        For Hierarchical Hybrid Enviroment which split one single action into many task level
    """
    def __init__(self, estimators: List[GraphPolicyBackbone], req_config: List,temperature: float = 1, sf_bias: float = 0, epsilon: float = 0, min_stop=3) -> None:
        super().__init__(estimators, req_config, temperature, sf_bias, epsilon)
        self.min_stop = min_stop

    def get_raw_logits(self, states, policy,subgoal):
        """
        Get logits from policy 
        """
        # logits = policy(states,subgoal)
        logits = policy(states, subgoal)
        return logits

    def get_logits(self,states, policy,subgoal=None):
        """
        Get logits from policy and and mask the illigel actions
        """
        logits = self.get_raw_logits(states, policy,subgoal)
        return logits

    def get_dist(self, states, policy):
        """
        Get distribution of the from the action
        """

        return policy.to_dist(states)
        
    def get_probs(self,states,policy,subgoal=None):
        """
        Get probs 
        """
        logits = self.get_logits(states,policy,subgoal)
        probs = [] # probs for each action types
        for logit in logits: #iterate all action types performed by an agent 
            prob = torch.softmax(logit / self.temperature, dim=-1)
            probs.append(prob)
        return probs, logits
    

    def adjust_lattice_based_on_action(self, lattices, sg):
        lattices[:,3:6] = torch.clamp(lattices[:,3:6],min=1.39626,max=2.26893)
        lattices[:,0:3] = torch.clamp(lattices[:,0:3],min=2.0)
        spacegroup_action = sg.clone().cpu().detach().numpy()
        spacegroup_action = np.squeeze(spacegroup_action)
        # spacegroup from model is 0-229
        for idx, sg in enumerate(spacegroup_action):
            if sg in [0, 1]:
                # print('Adjust to Triclinic')    
                if (lattices[idx,3] + lattices[idx,4] + lattices[idx,5] > 5.4):
                    lattices[idx,5] = math.pi/2
            elif sg in list(range(2, 15)):
                # print('adjust to Monoclinic')
                lattices[idx,3] = math.pi/2
                lattices[idx,5] = math.pi/2
            elif sg in list(range(15, 74)):
                # print('adjust to Orthorhombic')
                lattices[idx,3] = math.pi/2
                lattices[idx,4] = math.pi/2
                lattices[idx,5] = math.pi/2
            elif sg in list(range(74, 142)):
                # print('adjust to Tetragonal')
                lattices[idx,1] = lattices[idx,0]
                lattices[idx,3] = math.pi/2
                lattices[idx,4] = math.pi/2
                lattices[idx,5] = math.pi/2
            elif sg in list(range(142, 194)):
                # print('Adjust to hexagonal')
                lattices[idx,1] = lattices[idx,0]
                lattices[idx,3] = math.pi/2
                lattices[idx,4] = math.pi/2
                lattices[idx,5] = 2*math.pi/3
            elif sg in list(range(194, 230)):
                # print('Adjust to cubic')
                lattices[idx,1] = lattices[idx,0]
                lattices[idx,2] = lattices[idx,0]
                lattices[idx,3] = math.pi/2
                lattices[idx,4] = math.pi/2
                lattices[idx,5] = math.pi/2
            else:
                continue
        return lattices


    def sample(self,states,states_reps,
               n_trajectories,
               step=0,
               maxblock=3):
        # Array of actions from all level
        levels_action = []
        logprobs = 0

        probs_esg, logits_esg = self.get_probs(states=states_reps,policy=self.estimators[0])
        dists_esg = self.estimators[0].to_dist(probs_esg, logits_esg)
        exit_dist = dists_esg[0]
        sg_dist = dists_esg[1]

        with torch.no_grad():
            exit_action = exit_dist.sample()
            sg_action = sg_dist.sample()

        exit_logprobs = exit_dist.log_prob(exit_action)

        probs_la, logits_la = self.get_probs(states=states_reps,policy=self.estimators[1],subgoal=sg_action)

        mask = torch.tensor(np.array([state.get_valid_mask_atom_type(max_traj_len=maxblock) for state in states])).to(device=torch.device('cuda'))
        logits_la[13] = torch.where(mask, logits_la[13], torch.tensor(-float("inf")).to(device=torch.device('cuda')))
        dists_la = self.estimators[1].to_dist(probs_la, logits_la)
        l1_dist = dists_la[0]
        l2_dist = dists_la[1]
        l3_dist = dists_la[2]
        a1_dist = dists_la[3]
        a2_dist = dists_la[4]
        a3_dist = dists_la[5]
        frac_dist = dists_la[6]
        atomtype_dist = dists_la[7]

        with torch.no_grad():
            l1_action = l1_dist.sample()
            l2_action = l2_dist.sample()
            l3_action = l3_dist.sample()
            a1_action = a1_dist.sample()
            a2_action = a2_dist.sample()
            a3_action = a3_dist.sample()
            frac_action = frac_dist.sample()
            atype_action = atomtype_dist.sample()
        sg_logprobs = sg_dist.log_prob(sg_action)
        l1_logprobs = l1_dist.log_prob(l1_action)        
        l2_logprobs = l2_dist.log_prob(l2_action)        
        l3_logprobs = l3_dist.log_prob(l3_action)        
        a1_logprobs = a1_dist.log_prob(a1_action)        
        a2_logprobs = a2_dist.log_prob(a2_action)        
        a3_logprobs = a3_dist.log_prob(a3_action)        
        frac_logprobs = frac_dist.log_prob(frac_action)        
        atype_logprobs = atomtype_dist.log_prob(atype_action)

        lattice_action = torch.stack([l1_action, l2_action, l3_action, a1_action, a2_action, a3_action], dim=-1)
        lattice_action = self.adjust_lattice_based_on_action(lattice_action, sg_action)
        
        levels_action = [torch.unsqueeze(exit_action,1), torch.unsqueeze(sg_action,1), torch.squeeze(lattice_action), frac_action, torch.unsqueeze(atype_action,1)]
        action = torch.cat(levels_action, dim=1)
        logprobs += sg_logprobs + l1_logprobs + l2_logprobs + l3_logprobs + a1_logprobs + a2_logprobs + a3_logprobs + frac_logprobs + atype_logprobs
        if step > self.min_stop:
            action, logprobs = self.set_action_and_logprob_on_stop(action, logprobs, exit_logprobs, n_trajectories)
        return action, logprobs

    def set_action_and_logprob_on_stop(self, action, logprobs, exit_logprob, n_trajectories):
        terminated_mask = torch.where(action[...,0] == 1.0, True, False)
        action[terminated_mask] = torch.full(size=(int(terminated_mask.float().sum()),12), fill_value=-float("inf"), device=torch.device('cuda'))
        logprobs[terminated_mask] = exit_logprob[terminated_mask]
        return action, logprobs

class BackwardActionsSampler(HierarchicalActionSampler):
    """
    Base class for backward action sampling methods.
    """
    def __init__(self, estimators: List[GraphPolicyBackbone], req_config, temperature: float = 1, sf_bias: float = 0, epsilon: float = 0) -> None:
        super().__init__(estimators=estimators, req_config=req_config , temperature=temperature, sf_bias=sf_bias, epsilon=epsilon)

    def get_bw_dists(self,states_reps):
        # action [haction_1,haction_2,...,haction_n]
        dists= []
        for idx, p_level in enumerate(self.estimators):
            # Get probs and logits from the current state
            probs, logits = self.get_probs(states=states_reps,policy=p_level) 
            # Get distribution from probs (discret)/logits(continous)
            dist = p_level.to_dist(probs, logits) # [n_trajectories]
            dists.extend(dist)
        return dists