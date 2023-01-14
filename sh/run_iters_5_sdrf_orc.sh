python run_graph_classification.py --rewiring sdrf_orc --num_iterations 5 --layer_type GCN --num_trials 10 --device cuda:0
python run_graph_classification.py --rewiring sdrf_orc --num_iterations 5 --layer_type GIN --num_trials 10 --device cuda:0
python run_graph_classification.py --rewiring sdrf_orc --num_iterations 5 --layer_type R-GCN --num_trials 10 --device cuda:0
python run_graph_classification.py --rewiring sdrf_orc --num_iterations 5 --layer_type R-GIN --num_trials 10 --device cuda:0
