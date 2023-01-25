### GCN ###
python run_graph_classification.py --layer_type GCN --rewiring brf --num_trials 100 --num_iterations 1 --brf_batch_add 3 --brf_batch_remove 2 --dataset enzymes

python run_graph_classification.py --layer_type GCN --rewiring brf --num_trials 100 --num_iterations 1 --brf_batch_add 3 --brf_batch_remove 0 --dataset imdb

python run_graph_classification.py --layer_type GCN --rewiring brf --num_trials 100 --num_iterations 1 --brf_batch_add 20 --brf_batch_remove 3 --dataset mutag

python run_graph_classification.py --layer_type GCN --rewiring brf --num_trials 100 --num_iterations 3 --brf_batch_add 4 --brf_batch_remove 1 --dataset proteins

### GIN ###
python run_graph_classification.py --layer_type GIN --rewiring brf --num_trials 100 --num_iterations 1 --brf_batch_add 3 --brf_batch_remove 2 --dataset enzymes

python run_graph_classification.py --layer_type GIN --rewiring brf --num_trials 100 --num_iterations 1 --brf_batch_add 3 --brf_batch_remove 0 --dataset imdb

python run_graph_classification.py --layer_type GIN --rewiring brf --num_trials 100 --num_iterations 1 --brf_batch_add 20 --brf_batch_remove 3 --dataset mutag

python run_graph_classification.py --layer_type GIN --rewiring brf --num_trials 100 --num_iterations 3 --brf_batch_add 4 --brf_batch_remove 1 --dataset proteins
