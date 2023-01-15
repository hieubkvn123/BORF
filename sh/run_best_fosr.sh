python run_graph_classification.py --rewiring fosr --layer_type GCN --dataset imdb --num_iterations 5
python run_graph_classification.py --rewiring fosr --layer_type GCN --dataset mutag --num_iterations 40
python run_graph_classification.py --rewiring fosr --layer_type GCN --dataset enzymes --num_iterations 10
python run_graph_classification.py --rewiring fosr --layer_type GCN --dataset proteins --num_iterations 20

python run_graph_classification.py --rewiring fosr --layer_type GIN --dataset imdb --num_iterations 20
python run_graph_classification.py --rewiring fosr --layer_type GIN --dataset mutag --num_iterations 20
python run_graph_classification.py --rewiring fosr --layer_type GIN --dataset enzymes --num_iterations 5
python run_graph_classification.py --rewiring fosr --layer_type GIN --dataset proteins --num_iterations 10
