# BORF
python run_graph_classification_lrgb.py --dataset pt_func --rewiring borf --layer_type GCN --num_trials 16 --device cuda:0 --num_iterations 4 --borf_batch_add 30 --borf_batch_remove 20
python run_graph_classification_lrgb.py --dataset pt_func --rewiring borf --layer_type GIN --num_trials 16 --device cuda:0 --num_iterations 2 --borf_batch_add 40 --borf_batch_remove 10

# Default
python run_graph_classification_lrgb.py --dataset pt_func --rewiring None --layer_type GCN --num_trials 16 --device cuda:0
python run_graph_classification_lrgb.py --dataset pt_func --rewiring None --layer_type GIN --num_trials 16 --device cuda:0

# FoSR
python run_graph_classification_lrgb.py --dataset pt_func --rewiring fosr --layer_type GCN --num_trials 16 --device cuda:0 --num_iterations 25
python run_graph_classification_lrgb.py --dataset pt_func --rewiring fosr --layer_type GIN --num_trials 16 --device cuda:0 --num_iterations 150

# SDRF 
python run_graph_classification_lrgb.py --dataset pt_func --rewiring sdrf --layer_type GCN --num_trials 16 --device cuda:0 --num_iterations 100 --sdrf_remove_edges
python run_graph_classification_lrgb.py --dataset pt_func --rewiring sdrf --layer_type GIN --num_trials 16 --device cuda:0 --num_iterations 50 --sdrf_remove_edges
