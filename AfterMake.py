import json
import os.path
from os.path import join

def JsonSort():
    path = r'G:\qxz\DataSet\BrainSegDataSet301\DataSet4\Predict3\info.json'
    savePath = join(os.path.dirname(path), 'info_sort.json')
    with open(path, 'r') as f:
        data = json.loads(f.read())
        data = sorted(data, key=lambda x: x[-1].split(' ')[0])
        with open(savePath, 'w') as f:
            f.write(json.dumps(data))

if __name__ == '__main__':
    JsonSort()
