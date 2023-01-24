gpu_index = 0
layer_types = ['GCN', 'GIN']
num_iterations = [5, 10, 15, 20, 30, 40]
remove_edges = [True, False]

cmd_template = """
python run_graph_classification.py 
    --rewiring fosr 
    --layer_type {}
    --num_trials 100
    --device cuda:{}
    --num_iterations {}
"""

for layer_type in layer_types:
    for remove_edge in remove_edges:
        for iters in num_iterations:
            cmd = cmd_template.format(layer_type, gpu_index, iters)
            cmd = cmd.strip()
            cmd = cmd.replace('\n', '')
            cmd = cmd.replace('\t', ' ')

            if(remove_edge):
                cmd += ' --sdrf_remove_edges'

            print(cmd)
