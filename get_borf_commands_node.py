
gpu_index = 0
layer_types = ["GCN", "GIN"]
batch_add = [20, 30, 40]
batch_remove = [10, 20, 30]
num_iterations = [1, 2, 3]
datasets = ['actor', 'pubmed', 'squirrel'] # , 'ogbn-arxiv']

cmd_template = """
python run_node_classification.py 
    --rewiring borf 
    --layer_type {}
    --num_trials 100
    --device cuda:{}
    --borf_batch_add {}
    --borf_batch_remove {} 
    --num_iterations {}
    --dataset {}
"""

for layer in layer_types:
    for ba in batch_add:
        for br in batch_remove:
            for iters in num_iterations:
                for ds in datasets:
                    cmd = cmd_template.format(layer, gpu_index, ba, br, iters, ds)
                    cmd = cmd.strip()
                    cmd = cmd.replace('\n', '')
                    cmd = cmd.replace('\t', ' ')

                    print(cmd)
