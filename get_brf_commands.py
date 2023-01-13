
gpu_index = 0
layer_types = ["GCN", "GIN"]
batch_add = [3, 4, 5]
batch_remove = [1, 2, 3]
num_iterations = [1, 2, 3]

cmd_template = """
python run_graph_classification.py 
    --rewiring brf 
    --layer_type {}
    --num_trials 10
    --device cuda:{}
    --brf_batch_add {}
    --brf_batch_remove {} 
    --num_iterations {}
"""

for layer in layer_types:
    for ba in batch_add:
        for br in batch_remove:
            for iters in num_iterations:
                cmd = cmd_template.format(layer, gpu_index, ba, br, iters)
                cmd = cmd.strip()
                cmd = cmd.replace('\n', '')
                cmd = cmd.replace('\t', ' ')

                print(cmd)
