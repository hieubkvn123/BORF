import glob
import pandas as pd

for _file in glob.glob('graph_classification_*'):
    print('\nResult for ', _file)
    df = pd.read_csv(_file)

    result = df.groupby(['dataset', 'num_iterations']).max('test_mean')[['test_mean', 'test_ci']]
    result = result.sort_index(ascending=True)
    print(result)
