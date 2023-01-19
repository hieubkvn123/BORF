import pandas as pd

seeds = 10
gpu_index = 0
layer_type = 'GIN'
datasets = ['mutag', 'enzymes', 'proteins', 'imdb']
num_iterations = [1, 2, 3]
batch_add = [3, 4, 5]
batch_remove = [0, 1, 2, 3]

cmd_template = '''
python run_graph_classification.py --rewiring brf     
    --layer_type {}   
    --num_trials {}    
    --device cuda:{}    
    --brf_batch_add {} 
    --brf_batch_remove {}     
    --num_iterations {}
    --dataset {}
'''

exp_file = f'graph_classification_{layer_type}_brf.csv'
df = pd.read_csv(exp_file)
existing_exps = df[['dataset', 'num_iterations', 'brf_batch_add', 'brf_batch_remove']].values
existing_exps = [list(x) for x in existing_exps]
missing_exps = []
num_missing = 0

for ds in datasets:
    for it in num_iterations:
        for ba in batch_add:
            for br in batch_remove:
                df_ = df[df['dataset'] == ds]
                df_ = df_[df_['brf_batch_add'] == ba]
                df_ = df_[df_['brf_batch_remove'] == br]
                df_ = df_[df_['num_iterations'] == it]
                if(len(df_) == 0):
                    num_missing += 1
                    cmd = cmd_template.format(layer_type, seeds, gpu_index, ba, br, it, ds)
                    cmd = cmd.strip()
                    cmd = cmd.replace('\n', '')
                    cmd = cmd.replace('\t', ' ')
                    print(cmd)

print(f'# Number of missing experiments : {num_missing}')

