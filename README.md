# Multi-MoTSE - Julian Lee

This repo is built on top of the implementation for "MoTSE: an interpretable task similarity estimator for small molecular property prediction tasks" [[link]](https://github.com/lihan97/MoTSE/tree/main/src). 

Code was run using T4 GPUs on Google Colab. Dependencies are specified in requirements.txt. 

NOTE: saved_models.zip and datasets.zip need to be uncompressed.

## How to Run

Code is included to run the scratch, MoTSE, and multi-MoTSE models on the QM9 and PCBA datasets. To run the code, change the path_to_src variable in the first code block.

src/scratch.ipynb --> train scratch model
src/motse.ipynb --> train MoTSE model
src/multi-motse.ipynb --> train multi-MoTSE model (requires running the "Calculate Similarity" section of src/motse.ipynb first)
src/analysis.ipynb --> generate graphs of my results

## Folder + File Descriptions

datasets: stores QM9, PCBA, and probe (Zinc) datasets
saved_models: stores saved models
results: stores results 

In addition to src/multi-motse.ipynb, further implementation details of the multi-motse GNN can be found in src/multi_trainer.py, src/models.py (modified to support multi-task learning). 

Important results folders naming conventions: 
1. 1000 means scratch
2. 10000->1000 means MoTSE (testing transfers from 3 most similar datasets)
3. multi10000-3 means multi-MoTSE (using 3 most similar datasets)
