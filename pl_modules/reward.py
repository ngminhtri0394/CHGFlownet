from metrics.eval_metrics import *
from common.bonds_dictionary import bonds_dictionary

def get_bs_dm_min_max(m: Structure, vpen_min=0.01,vpen_max=0.1,vpen_minmax=0.001):
    bond_min_atm = bonds_dictionary['min']
    bond_avg_atom = bonds_dictionary['mean']
    bond_std_atom = bonds_dictionary['std']

    a = m.atomic_numbers
    dist = m.distance_matrix
    nbond = 0
    vmin = 0
    vmax = 0 
    for i in range(0,len(a)-1):
        has_near_neighbour = False 
        for j in range(i+1,len(a)):
            nbond +=1
            bond_name = str(a[i]) + '_' + str(a[j])
            if bond_name not in bond_min_atm:
                bond_name = str(a[j]) + '_' + str(a[i])
                if bond_name not in bond_min_atm:
                    min_constraint = 1.0
                    neighbour_constraint = 3.5
                    # there is no bond dist data in the dictionary, use default value
                else:
                    min_constraint = bond_min_atm[bond_name]
                    neighbour_constraint = bond_avg_atom[bond_name] + bond_std_atom[bond_name]
            else:
                min_constraint = bond_min_atm[bond_name]
                neighbour_constraint = bond_avg_atom[bond_name] + bond_std_atom[bond_name]

            if dist[i][j] < min_constraint:
                vmin+=1
            if has_near_neighbour == False:
                if dist[i][j] < neighbour_constraint:
                    has_near_neighbour = True
        if has_near_neighbour == False:
            vmax+= 1
    vmin = (nbond-vmin)/nbond
    vmax = (len(a) - vmax)/len(a)
    if vmin < 1 and vmax < 1:
        return vpen_minmax
    if vmin < 1:
        return vmin*vpen_min
    if vmax < 1:
        return vmax*vpen_max
    return 1.0

def get_bs_min(m: Structure, vpen_min=0.01,vpen_max=0.1,vpen_minmax=0.001):
    a = m.atomic_numbers
    dist = m.distance_matrix
    nbond = 0
    vmin = 0
    min_constraint = 0.5
    for i in range(0,len(a)-1):
        for j in range(i+1,len(a)):
            nbond +=1
            if dist[i][j] < min_constraint:
                vmin+=1
    vmin = (nbond-vmin)/nbond
    if vmin < 1:
        return vmin*vpen_min
    return 1.0

def get_valid_score(structure):
    atom_types = [s.specie.Z for s in structure]
    elems, comps = get_composition(atom_types)
    return max(float(smact_validity(elems, comps)),0.0)

def reward_func_gd_dm_max_dist(states, proxy, vpen_min=0.01, vpen_max=0.1, vpen_minmax=0.001):
    pred = proxy(states)
    score = np.exp(-pred)
    mean_density = 3.0
    std_density = 1.0
    alpha_density = 3
    bond_score = np.array([get_bs_dm_min_max(state.structure, vpen_min, vpen_max, vpen_minmax) for state in states])
    density_score = np.array([alpha_density*np.exp(-(m.structure.density-mean_density)**2/(2*(std_density**2))) for m in states])  
    reward = (score+density_score)*bond_score
    reward = np.clip(reward, 0.0001, 10.0)
    return reward, bond_score


def reward_func_gd_dm_max_dist_SMACT(states, proxy, vpen_min=0.01, vpen_max=0.1, vpen_minmax=0.001):
    pred = proxy(states)
    score = np.exp(-pred)
    mean_density = 3.0
    std_density = 1.0
    alpha_density = 3
    bond_score = np.array([get_bs_dm_min_max(state.structure, vpen_min, vpen_max, vpen_minmax) for state in states])
    density_score = np.array([alpha_density*np.exp(-(m.structure.density-mean_density)**2/(2*(std_density**2))) for m in states])  
    # density_score = np.array([m.structure.density for m in states])  
    valid_score = np.array([get_valid_score(state.structure) for state in states])
    reward = (score+density_score)*bond_score + valid_score
    reward = np.clip(reward, 0.0001, 10.0)
    return reward, bond_score


def reward_func_gd_dm_max_dist_no_forme(states, proxy, vpen_min=0.01, vpen_max=0.1, vpen_minmax=0.001):
    mean_density = 3.0
    std_density = 1.0
    alpha_density = 3
    bond_score = np.array([get_bs_dm_min_max(state.structure, vpen_min, vpen_max, vpen_minmax) for state in states])
    density_score = np.array([alpha_density*np.exp(-(m.structure.density-mean_density)**2/(2*(std_density**2))) for m in states])  
    # density_score = np.array([m.structure.density for m in states])  
    valid_score = np.array([get_valid_score(state.structure) for state in states])
    reward = (density_score)*bond_score + valid_score
    reward = np.clip(reward, 0.0001, 10.0)
    return reward, bond_score


def reward_func_gd_dm_max_dist_SMACT_nodensity(states, proxy, vpen_min=0.01, vpen_max=0.1, vpen_minmax=0.001):
    pred = proxy(states)
    score = np.exp(-pred)
    mean_density = 3.0
    std_density = 1.0
    alpha_density = 3
    bond_score = np.array([get_bs_dm_min_max(state.structure, vpen_min, vpen_max, vpen_minmax) for state in states])
    valid_score = np.array([get_valid_score(state.structure) for state in states])
    reward = (score)*bond_score + valid_score
    reward = np.clip(reward, 0.0001, 10.0)
    return reward, bond_score


def reward_func_gd_dm_max_dist_SMACT_only_min(states, proxy, vpen_min=0.01, vpen_max=0.1, vpen_minmax=0.001):
    pred = proxy(states)
    score = np.exp(-pred)
    mean_density = 3.0
    std_density = 1.0
    alpha_density = 3
    bond_score = np.array([get_bs_min(state.structure, vpen_min, vpen_max, vpen_minmax) for state in states])
    density_score = np.array([alpha_density*np.exp(-(m.structure.density-mean_density)**2/(2*(std_density**2))) for m in states])  
    valid_score = np.array([get_valid_score(state.structure) for state in states])
    reward = (score + density_score)*bond_score + valid_score
    reward = np.clip(reward, 0.0001, 10.0)
    return reward, bond_score


reward_functions_dict = {'reward_func_gd_dm_max_dist_SMACT_only_min':reward_func_gd_dm_max_dist_SMACT_only_min,
                         'reward_func_gd_dm_max_dist_SMACT_nodensity':reward_func_gd_dm_max_dist_SMACT_nodensity,
                         'reward_func_gd_dm_max_dist_no_forme':reward_func_gd_dm_max_dist_no_forme,
                         'reward_func_gd_dm_max_dist_SMACT':reward_func_gd_dm_max_dist_SMACT}
