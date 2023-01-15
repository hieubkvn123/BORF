import glob
import pandas as pd

model = 'brf'

for _file in glob.glob(f'graph_classification_*{model}*'):
    print('\nResult for ', _file)
    df = pd.read_csv(_file)

    if(model == 'brf'):
        result = df.groupby(['dataset', 'num_iterations']).max('test_mean')[['test_mean', 'test_ci', 'brf_batch_add', 'brf_batch_remove']]
    else:
        result = df.groupby(['dataset', 'num_iterations']).max('test_mean')[['test_mean', 'test_ci']]

    result = result.sort_index(ascending=True)
    print(result)
    
    print(f'Saving to summary_{_file}.xlsx')
    result.to_excel(f'summary_{_file}.xlsx')
