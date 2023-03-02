from argparse import ArgumentParser

def get_commands(dataset, gpu_index, hyperparams):
    # Unpack hyper-parameters
    if(isinstance(dataset, list)):
        datasets = dataset
    else:
        datasets = [dataset]
    layer_types = ['GCN', 'GIN']
    batch_add = hyperparams['batch_add']
    batch_remove = hyperparams['batch_remove']
    num_iterations = hyperparams['num_iterations']

    # Output commands
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

    for ds in datasets:
        for layer in layer_types:
            for ba in batch_add:
                for br in batch_remove:
                    for iters in num_iterations:
                        cmd = cmd_template.format(layer, gpu_index, ba, br, iters, ds)
                        cmd = cmd.strip()
                        cmd = cmd.replace('\n', '')
                        cmd = cmd.replace('\t', ' ')

                        print(cmd)

# Constants
gpu_index = 0
datasets = ['actor', 'pubmed', 'squirrel', 'ogbn-arxiv'] 

# Define multiple grids corresponding to multiple tuning strategies
hyperparams_grid = {
    0 : {
        'batch_add' : [20, 30, 40],
        'batch_remove' : [10, 20, 30],
        'num_iterations' : [1, 2, 3]
    },
    1 : {
        'batch_add' : [2, 3, 4],
        'batch_remove' : [1, 2, 3],
        'num_iterations' : [5, 10, 15, 20, 30]
    }
}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default=None, required=False)
    parser.add_argument('--gpu_index', default=0, required=False)
    args = vars(parser.parse_args())

    if(args['dataset']):
        datasets = [args['dataset']]
    gpu_index = args['gpu_index']

    for i, hyperparams in hyperparams_grid.items():
        get_commands(datasets, gpu_index, hyperparams)
