# python run_graph_classification.py --rewiring sdrf_orc --num_iterations 10 --layer_type R-GIN --num_trials 10
python run_graph_classification.py --rewiring sdrf_bfc --num_iterations 10 --layer_type R-GIN --num_trials 10
python run_graph_classification.py --rewiring sdrf_orc --num_iterations 10 --layer_type R-GCN --num_trials 10
python run_graph_classification.py --rewiring sdrf_bfc --num_iterations 10 --layer_type R-GCN --num_trials 10
