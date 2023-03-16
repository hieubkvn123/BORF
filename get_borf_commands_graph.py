import os
from argparse import ArgumentParser

def get_commands(gpu_index, num_trials, datasets, layer_types, batch_add,
    batch_remove, num_iterations):
    cmd_template = """
        python run_graph_classification_lrgb.py 
            --rewiring borf 
            --layer_type {}
            --num_trials {}
            --device cuda:{}
            --borf_batch_add {}
            --borf_batch_remove {} 
            --num_iterations {}
            --dataset {}
    """
    commands = []
    for layer in layer_types:
        for ba in batch_add:
            for br in batch_remove:
                for iters in num_iterations:
                    for ds in datasets:
                        cmd = cmd_template.format(layer, num_trials, gpu_index, ba, br, iters, ds)
                        cmd = cmd.strip()
                        cmd = cmd.replace('\n', '')
                        cmd = cmd.replace('\t', ' ')
                        commands.append(cmd)

    return commands

if __name__ == '__main__':
    # User options
    parser = ArgumentParser()
    parser.add_argument('--gpu_index', type=int, default=0, required=False)
    parser.add_argument('--num_trials', type=int, default=100, required=False)
    parser.add_argument('--dataset', type=str, default='pt_func', required=False)
    parser.add_argument('--segments', type=int, default=4, required=False)
    parser.add_argument('--dir', type=str, default='scripts/rebuttals', required=False)
    args = vars(parser.parse_args())

    # Parse options
    datasets = [args['dataset']]
    gpu_index = args['gpu_index']
    num_trials = args['num_trials']
    save_dir = args['dir']

    # Hard-coded options
    layer_types = ["GCN", "GIN"]
    batch_add = [30]
    batch_remove = [20]
    num_iterations = [1, 2, 3, 4, 5, 6, 7, 8]

    # Get commands
    commands = get_commands(gpu_index, num_trials, datasets, layer_types, batch_add,
            batch_remove, num_iterations)
    n = len(commands) // args['segments']
    
    # Segment and write
    for i in range(0, len(commands), n):
        segment = commands[i:i + n]
        with open(os.path.join(save_dir, f'run_borf_{args["dataset"]}_{i//n + 1}.sh'), 'w') as f:
            for cmd in segment:
                f.write(cmd + '\n')
