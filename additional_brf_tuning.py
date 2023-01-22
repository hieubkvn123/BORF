
gpu_index = 0
idx = 0
batch_add = [20, 15, 10, 5]
batch_remove = [0, 1, 3, 6, 9]
num_iterations = [1, 2, 3]

cmd_template = """
python run_graph_classification.py 
    --rewiring brf 
    --layer_type {}
    --num_trials 100
    --device cuda:{}
    --brf_batch_add {}
    --brf_batch_remove {} 
    --num_iterations {}
    --dataset {}
"""

settings_of_interest = [
    {'dataset' : 'proteins', 'layer_type' : 'GIN'},
    {'dataset' : 'enzymes', 'layer_type' : 'GCN'},
    {'dataset' : 'mutag', 'layer_type' : 'GCN'},
    {'dataset' : 'proteins', 'layer_type' : 'GCN'}
]

setting = settings_of_interest[idx]
dataset, layer_type = setting['dataset'], setting['layer_type']

for ba in batch_add:
    for br in batch_remove:
        for iters in num_iterations:
            cmd = cmd_template.format(layer_type, gpu_index, ba, br, iters, dataset)
            cmd = cmd.strip()
            cmd = cmd.replace('\n', '')
            cmd = cmd.replace('\t', ' ')

            print(cmd)
