# RioGNN

Code for [**Reinforced Neighborhood Selection Guided Multi-Relational Graph Neural Networks**](https://arxiv.org/pdf/2104.07886.pdf).  

[Hao Peng](https://penghao-buaa.github.io/), Ruitong Zhang, [Yingtong Dou](http://ytongdou.com/), Renyu Yang, Jingyi Zhang, [Philip S. Yu](https://www.cs.uic.edu/PSYu/).


## Repo Structure

The repository is organized as follows:
- `data/`: dataset folder
    - `YelpChi.zip`: Data of the dataset Yelp;
    - `Amazon.zip`: Data of the dataset Amazon;
    - `Mimic.zip`: Data of the dataset Mimic;
- `log/`: log folder
- `model/`: model folder
    - `graphsage.py`: model code for vanilla [GraphSAGE](https://github.com/williamleif/graphsage-simple/) model;
    - `layers.py`: RioGNN layers implementations;
    - `model.py`: RioGNN model implementations;
- `RL/`: RL folder
    - `actor_critic.py`: RL algorithm, [Actor-Critic](https://github.com/llSourcell/actor_critic);
    - `rl_model.py`: RioGNN RL Forest implementations;
- `utils/`: functions folder
    - `data_process.py`: transfer sparse matrix to adjacency lists;
    - `utils.py`: utility functions for data i/o and model evaluation;
- `train.py`: training and testing all models


## Example Dataset

We build different multi-relational graphs for experiments in two task scenarios and three datasets: 

| Dataset  | Task  | Nodes  | Relation  |
|-------|--------|--------|--------|
| Yelp  | Fraud Detection | 45,954  | rur, rtr, rsr, homo |
| Amazon  | Fraud Detection | 11,944  | upu, usu, uvu, homo |
| MIMIC-III  | Diabetes Diagnosis | 28,522  | vav, vdv, vpv, vmv, homo |

## Run on your Datasets

To run RioGNN on your datasets, you need to prepare the following data:

- Multiple-single relation graphs with the same nodes where each graph is stored in `scipy.sparse` matrix format, you can use `sparse_to_adjlist()` in `utils.py` to transfer the sparse matrix into adjacency lists used by RioGNN;
- A numpy array with node labels. Currently, RioGNN only supports binary classification;
- A node feature matrix stored in `scipy.sparse` matrix format. 


## How to Run
You can download the project and and run the program as follows:

###### 1. The dataset folder `\data` ontains LFS file `Mimic.zip`, please use the explicit command `git lfs clone`;
```bash
git lfs clone git@github.com:safe-graph/RioGNN.git
```
Or, you can also clone other files first by command, and then download the dataset `Mimic.zip` (734.5MB) via the link below;
```bash
git clone git@github.com:safe-graph/RioGNN.git
```
https://drive.google.com/XXXXX 

\* Note that all datasets need to be unzipped in the folder `\data` first;
###### 2. Install the required packages using the `requirements.txt`;
```bash
pip3 install -r requirements.txt
```
###### 3. Run `data_process.py` to generate adjacency lists of different dataset used by RioGNN;
```bash
python data_process.py
```
###### 4. Run `train.py` to run RioGNN with default settings.
```bash
python train.py
```

\* To run the code, you need to have at least **Python 3.6** or later versions. 

## Important Parameters

- Our model supports both CPU and GPU mode, you can change it through parameter `--use_cuda` and  `--device`:
- Set the `--data` as `yelp`, `amazon` or `mimic` to change different dataset.
- Parameter `--num_epochs` is used to set the maximum number of iterative epochs. 
Note that the model will stop early when reinforcement learning has explored all depths.
- The default value of parameter `--ALAPHA` is `10`, 
which means that the accuracy of different depths of reinforcement learning tree will be progressive with 
0.1, 0.01, 0.001, etc. 
If you want to conduct more width and depth experiments, please adjust here.

\* For other dataset and parameter settings, please refer to the arg parser in `train.py`. 


## Preliminary Work

Our preliminary work, **CA**mouflage-**RE**sistant **G**raph **N**eural **N**etwork 
**([CARE-GNN](https://github.com/YingtongDou/CARE-GNN))**,
is a GNN-based fraud detector based on a multi-relation graph equipped with three modules that 
enhance its performance against camouflaged fraudsters.




## Citation
If you use our code, please cite the paper below:
```bibtex
@article{peng2021reinforced,
  title={Reinforced Neighborhood Selection Guided Multi-Relational Graph Neural Networks},
  author={Peng, Hao and Zhang, Ruitong and Dou, Yingtong and Yang, Renyu and Zhang, Jingyi and Yu, Philip S.},
  journal={ACM Transactions on Information Systems (TOIS)},
  year={2021}
}
```
