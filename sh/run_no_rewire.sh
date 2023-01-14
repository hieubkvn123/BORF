python run_graph_classification.py --rewiring none --num_iterations 3 --layer_type GCN --num_trials 10 --device cuda:0
python run_graph_classification.py --rewiring none --num_iterations 3 --layer_type GIN --num_trials 10 --device cuda:0
python run_graph_classification.py --rewiring none --num_iterations 5 --layer_type GCN --num_trials 10 --device cuda:0
python run_graph_classification.py --rewiring none --num_iterations 5 --layer_type GIN --num_trials 10 --device cuda:0
python run_graph_classification.py --rewiring none --num_iterations 10 --layer_type GCN --num_trials 10 --device cuda:0
python run_graph_classification.py --rewiring none --num_iterations 10 --layer_type GIN --num_trials 10 --device cuda:0
