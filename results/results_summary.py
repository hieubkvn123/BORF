import glob
import pandas as pd

model = 'brf'
keys = ['dataset', 'num_iterations']
columns = ['avg_accuracy', 'ci', 'brf_batch_add', 'brf_batch_remove']
if(model != 'brf'):
    columns = ['avg_accuracy', 'ci']
columns = keys + columns

for _file in glob.glob(f'node_classification_*{model}*'):
    print('\nResult for ', _file)
    df = pd.read_csv(_file)

    if(model == 'none'):
        df = df[df['num_iterations'] == 10]

    # Find idx of max accuracy
    idx = df.groupby(['dataset', 'num_iterations']).idxmax(numeric_only=True)['avg_accuracy']
    result = df.loc[idx][columns]
    result = result.groupby(['dataset', 'num_iterations']).max()
    print(result)
    
    print(f'Saving to summary_{_file}.xlsx')
    result.to_excel(f'summary_{_file}.xlsx')
