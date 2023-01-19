python run_graph_classification.py --rewiring brf --layer_type GIN --num_trials 100  --device cuda:0 --brf_batch_add 3 --brf_batch_remove 1 --num_iterations 3 --dataset enzymes
python run_graph_classification.py --rewiring brf --layer_type GIN --num_trials 100  --device cuda:0 --brf_batch_add 4 --brf_batch_remove 2 --num_iterations 1 --dataset imdb
python run_graph_classification.py --rewiring brf --layer_type GIN --num_trials 100  --device cuda:0 --brf_batch_add 3 --brf_batch_remove 1 --num_iterations 1 --dataset mutag
python run_graph_classification.py --rewiring brf --layer_type GIN --num_trials 100  --device cuda:0 --brf_batch_add 4 --brf_batch_remove 3 --num_iterations 2 --dataset proteins
