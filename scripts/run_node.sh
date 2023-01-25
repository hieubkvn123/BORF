### GCN ###
python run_node_classification.py --layer_type GCN --rewiring brf --num_trials 100 --num_iterations 3 --brf_batch_add 20 --brf_batch_remove 10 --dataset cora 

python run_node_classification.py --layer_type GCN --rewiring brf --num_trials 100 --num_iterations 3 --brf_batch_add 20 --brf_batch_remove 10 --dataset citeseer 

python run_node_classification.py --layer_type GCN --rewiring brf --num_trials 100 --num_iterations 3 --brf_batch_add 30 --brf_batch_remove 10 --dataset texas 

python run_node_classification.py --layer_type GCN --rewiring brf --num_trials 100 --num_iterations 2 --brf_batch_add 20 --brf_batch_remove 30 --dataset cornell 

python run_node_classification.py --layer_type GCN --rewiring brf --num_trials 100 --num_iterations 2 --brf_batch_add 30 --brf_batch_remove 20 --dataset wisconsin 

python run_node_classification.py --layer_type GCN --rewiring brf --num_trials 100 --num_iterations 3 --brf_batch_add 20 --brf_batch_remove 20 --dataset chameleon 

### GIN ###
python run_node_classification.py --layer_type GIN --rewiring brf --num_trials 100 --num_iterations 3 --brf_batch_add 20 --brf_batch_remove 10 --dataset cora 

python run_node_classification.py --layer_type GIN --rewiring brf --num_trials 100 --num_iterations 3 --brf_batch_add 20 --brf_batch_remove 10 --dataset citeseer 

python run_node_classification.py --layer_type GIN --rewiring brf --num_trials 100 --num_iterations 3 --brf_batch_add 30 --brf_batch_remove 10 --dataset texas 

python run_node_classification.py --layer_type GIN --rewiring brf --num_trials 100 --num_iterations 2 --brf_batch_add 20 --brf_batch_remove 30 --dataset cornell 

python run_node_classification.py --layer_type GIN --rewiring brf --num_trials 100 --num_iterations 2 --brf_batch_add 30 --brf_batch_remove 20 --dataset wisconsin 

python run_node_classification.py --layer_type GIN --rewiring brf --num_trials 100 --num_iterations 3 --brf_batch_add 20 --brf_batch_remove 20 --dataset chameleon 
