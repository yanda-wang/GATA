# GATA
The source code for the paper "Graph Augmented Triplet Architecture for Fine-Grained Patient Similarity"

This is the implement of Graph Augmented Triplet Architecture (GATA), a framework proposed in "Graph Augmented Triplet Architecture for Fine-Grained Patient Similarity" that aims to learning fine-grained patient similarity based on EHRs. GATA firstly derives Dynamic Bayesian Networks (DBNs) from EHRs to reveal correlations among medical variables, then it constructs graph augmented RNNs where each unit aggregate information from variables that it conditionally dependent in DBNs. After that, the specially designed RNNs will act as the fundamental components of the Triplet architecture to measure similarities among patients.

The framework is implemented with PyTorch. Codes in Auxiliary.py carries out preprocessing on EHRs to convert raw data to the form that required by GATA as well as Med2Vec, a framework we use to obtain meaningful embedding vectors of medical concepts. Codes in SupervisedInformation.py generate triplets of patients to act as inputs of GATA with supervised patient similarity calculated. Codes in Triplet.py implement the graph augmented RNNs according to DBNs derived from ERHs, and use them as fundamental modules of a Triplet archiecture for fine-grained patient similarity learning.

You could find more details and descriptions of the framwork in the paper.
