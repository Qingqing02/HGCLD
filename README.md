
## HGCLD
 Hierarchical Graph Contrastive Learning architecture with Diffusion-enhanced for multi-behavior
recommendation

## 🚀 Abstract

Multi-behavior recommendation (leverages user-item interaction data across multiple behaviors to gain a deeper understanding of user preferences and enhance recommendation performance.) This paper presents a Hierarchical Graph Contrastive Learning architecture with Diffusionenhanced (HGCLD) for multi-behavior recommendation. Specifically, our HGCLD first designs a hierarchical behavior-aware graph diffusion model with a cross-type semantic transition strategy for constructing hierarchical contrastive views. In essence, each auxiliary behavior is individually transformed into target semantic spaces through diffusion models for generating contrastive views. This process not only enhances the alignment between auxiliary behavior features and target semantic spaces but also effectively prevents the propagation of auxiliary behavior-specific noise to other auxiliary behavior embeddings. The hierarchical contrastive views then are utilised as anchors to align user behaviour patterns across different contrastive views, optimizing the main objective functions for model parameter updates and thus improving the learning of behavior-aware user representations. We conduct comprehensive experiments on three multi-behavior datasets, demonstrating the effectiveness of HGCLD and its components compared to various state-of-the-art methods.

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

