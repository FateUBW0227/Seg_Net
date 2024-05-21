import json, os
from os.path import join
import numpy as np

def MeanPR():
    path = r'G:\qxz\DataSet\BrainSegDataSet301\HaiNanDataSet1\DataSet128\Predict\info.json'
    with open(path, 'r') as f:
        data = json.loads(f.read())
        prs = []
        for item in data:
            prs.append(np.array(item[-1].split(' '), dtype=np.float32))
        prs = np.array(prs)
        meanPrs = np.mean(prs, axis=0)
        print(meanPrs)

if __name__ == '__main__':
    MeanPR()
