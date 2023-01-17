# FoSR
python run_graph_classification.py --rewiring fosr --layer_type GCN --dataset collab --num_iterations 10
python run_graph_classification.py --rewiring fosr --layer_type GCN --dataset reddit --num_iterations 5
python run_graph_classification.py --rewiring fosr --layer_type GIN --dataset collab --num_iterations 20
python run_graph_classification.py --rewiring fosr --layer_type GIN --dataset reddit --num_iterations 10

# SDRF
python run_graph_classification.py --rewiring sdrf --layer_type GCN --dataset collab --num_iterations 5
python run_graph_classification.py --rewiring sdrf --layer_type GCN --dataset reddit --num_iterations 5
python run_graph_classification.py --rewiring sdrf --layer_type GIN --dataset collab --num_iterations 40
python run_graph_classification.py --rewiring sdrf --layer_type GIN --dataset reddit --num_iterations 5
