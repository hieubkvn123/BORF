import glob
import pandas as pd

model = ['borf', 'sdrf', 'None']
keys = ['dataset', 'num_iterations']
columns = ['avg_accuracy', 'ci', 'borf_batch_add', 'borf_batch_remove']
if(model != 'borf'):
    columns = ['avg_accuracy', 'ci']
columns = keys + columns

if(isinstance(model, list)):
    models = model
else:
    models = [model]
for m in models:
    for _file in glob.glob(f'node_classification_*{m}*'):
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
