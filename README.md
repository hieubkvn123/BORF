# BORF
BORF: Batch Ollivier Ricci Flow for unifying and addressing over-smoothing and over-squashing in GNN. 

## Requirements
To configure and activate the conda environment for this repository, run
```
conda env create -f environment.yml
conda activate borf 
pip install -r requirements.txt
```

## Experiments
### 1. For graph classification
To run experiments for the TUDataset benchmark, run the file ```run_graph_classification.py```. The following command will run the benchmark for BORF with 20 iterations:
```bash
python run_graph_classification.py --rewiring brf --num_iterations 20
```

To add options for number of edges added and removed for rewiring, add the --brf_batch_add and --brf_batch_remove options
```bash
# Runs BORF with 3 batches, add 3 edges per batch and remove 1 edge per batch
python run_graph_classification.py --rewiring brf --num_iterations 3 \
	--brf_batch_add 3 \
	--brf_batch_remove 1
```

### 2. For node classification
To run node classification, simply change the script name to `run_node_classification.py`. For example:
```bash
python run_node_classification.py --rewiring brf --num_iterations 3 \
	--brf_batch_add 3 \
	--brf_batch_remove 1
```
