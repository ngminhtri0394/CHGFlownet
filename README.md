# Hierarchical GFlownet for Crystal Structure Generation

This is the Pytorch implementation of Hierarchical Generative Flow Network (CHGlownet), a new generative model that employs a hierarchical exploration strategy with Generative Flow Network to efficiently explore the material space while generating the crystal structure with desired properties. Our model decomposes the large material space into a hierarchy of subspaces of space groups, lattice parameters, and atoms.

# Usage 

First, create conda environment and active the :
```
conda env create --file=environment.yml
conda activate CHGFlownet
```
Run CHGFlownet:

```
python train_CHGFlownet.py
```

# Citation

If you find this work useful, please cite our paper:
```
@inproceedings{nguyen2023hierarchical,
  title={Hierarchical GFlowNet for Crystal Structure Generation},
  author={Nguyen, Tri Minh and Tawfik, Sherif Abdulkader and Tran, Truyen and Gupta, Sunil and Rana, Santu and Venkatesh, Svetha},
  booktitle={AI for Accelerated Materials Design-NeurIPS 2023 Workshop},
  year={2023}
}
```

