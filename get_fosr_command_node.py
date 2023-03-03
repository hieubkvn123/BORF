gpu_index = 0
layer_types = ["GCN", "GIN"]
num_iterations = [25, 50, 75, 100, 125, 150, 175, 200] 
datasets = ['pubmed', 'squirrel', 'ogbn-arxiv'] 

cmd_template = """
python run_node_classification.py 
    --rewiring fosr 
    --layer_type {}
    --num_trials 100
    --device cuda:{}
    --num_iterations {}
    --dataset {}
"""

for layer in layer_types:
    for iters in num_iterations:
        for ds in datasets:
            cmd = cmd_template.format(layer, gpu_index, iters, ds)
            cmd = cmd.strip()
            cmd = cmd.replace('\n', '')
            cmd = cmd.replace('\t', ' ')

            print(cmd)
