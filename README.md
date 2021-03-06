# MM-GNN

Repo with code associated with paper "Multimodal Graph Coarsening for Interpretable, MRI-Based Graph Neural Network"

MMSAG.py and MMTopK.py provide implementations of the multimodal SAGPooling and TopK pooling layers described in the paper for M=2 input modalities.

MMGNN.py provides code for the model architecture used to obtain the results on the COBRE case/control dataset, accessible [here](http://fcon_1000.projects.nitrc.org/indi/retro/cobre.html). 

The model can be integrated into a standard PyTorch training script using data formatted with the [PairData](https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html) function provided by PyTorch Geometric.


