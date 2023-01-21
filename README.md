# BORF
BORF: Batch Ollivier Ricci Flow for unifying and addressing over-smoothing and over-squashing in GNN. 

## Requirements
To configure and activate the conda environment for this repository, run
```
conda env create -f environment.yml
conda activate oversquashing
pip install -r requirements.txt
```

## Experiments
### 1. For graph classification
To run experiments for the TUDataset benchmark, run the file ```run_graph_classification.py```. The following command will run the benchmark for BORF with 20 iterations:
```
python run_graph_classification.py --rewiring brf --num_iterations 20
```

To add options for number of edges added and removed for rewiring, add the --brf_


