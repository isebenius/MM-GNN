# MM-GNN

Repo with code associated with paper "Multimodal Brain Explainer: Integrating Connectivity Data for Schizophrenia Detection"

MMSAG.py and MMTopK.py provide implementations of the multimodal SAGPooling and TopK pooling layers described in the paper for M=2 input modalities. \

Architecture.py provides code for the model architecture used to obtain the results on the COBRE case/control dataset, accessible [here](http://fcon_1000.projects.nitrc.org/indi/retro/cobre.html). 

Input to the model must be formatted using the [PairData](https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html) function provided by PyTorch Geometric.


