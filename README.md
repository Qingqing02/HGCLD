
## HGCLD
 Hierarchical Graph Contrastive Learning architecture with Diffusion-enhanced for multi-behavior
recommendation
## 📝 Environment

We develop our codes in the following environment:

- python=3.8
- torch=1.12.1
- numpy=1.23.1
- scipy=1.9.1
- dgl=1.0.2+cu113

## 📚 Datasets

| Datasets          | User        | Item      | Link | Interaction Types     |
| ------------------- | --------------- |-----------| ------------- |---------------|
|Retail Rocket            |2174      |  30113  |97381          | View, Cart, Transaction |
| Tmall            | 31882        | 31232   | 145129        | View, Favorite, Cart, Purchase       |
| IJCAI       |17435      | 35920 | 799368       |View, Favorite, Cart, Purchase       |

## 🚀 How to run the codes

In order to reproduce the results of HGCLD model on the datasets,you can kindly run the following command on Retail Rocket datasets for an instance.Before run the codes,you need to create the History and Models folders, and then create a retail_rocket folder in each of these two folders.The command lines to train HGCLD on the Retail Rocket datasets are as below, The un-specified hyperparameters in the commands are set as default.

- Retail Rocket 

```python
 python Main.py --data retail_rocket --gcn_layer 3 --contrast_weight 0.1 --steps 200 --batch 2048 --latdim 128 

```

## 👉 Code Structure

```
.
├── README.md
├── main.py
├── Model.py
├── params.py
├── DataHandler.py
├── Utils
│   ├── TimeLogger.py
│   └── Utils.py
└── data
    ├──ijcai_15 
    │   ├── test_mat.pkl
    │   ├── train_mat_buy.pkl
    |   ├── train_mat_cart.pkl
    |   ├── train_mat_click.pkl
    |   └── train_mat_fav.pkl 
    ├── retail_rocket
    │   ├── test_mat.pkl
    │   ├── train_mat_buy.pkl
    |   ├── train_mat_cart.pkl
    |   └── train_mat_view.pkl  
    └── tmall
        ├── test_mat.pkl
        ├── train_mat_buy.pkl
        ├── train_mat_cart.pkl
        ├── train_mat_fav.pkl
        └── train_mat_pv.pkl
```

## Acknowledgements
We are particularly grateful to the authors of DiffGraph, as parts of our code implementation were derived from their work. We have cited the relevant references in our paper.

