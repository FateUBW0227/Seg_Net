import json
import os
from os.path import join

def GenerateJsonV1():
    maskPath = r'E:\NeronSegDataSet\HaiNanDataSet1\DataSet128Small\smallMask'
    imgPath = r'E:\NeronSegDataSet\HaiNanDataSet1\DataSet128Small\images'
    savePath = r'E:\NeronSegDataSet\HaiNanDataSet1\DataSet128Small\smallMask.json'
    ls = os.listdir(maskPath)
    jsonInfo = []
    for name in ls:
        jsonInfo.append([join(maskPath, name), join(imgPath, name), ''])
    with open(savePath, 'w') as f:
        f.write(json.dumps(jsonInfo))

def GenerateJsonV2():
    rootPath = r'E:\NeronSegDataSet\HaiNanDataSet1\OriDataSet'
    imgDirName = 'images'
    maskDirName = 'mask'
    savePath = join(rootPath, imgDirName + '.json')
    ls = os.listdir(join(rootPath, maskDirName))
    jsonInfo = []
    for name in ls:
        jsonInfo.append([join(maskDirName, name), join(imgDirName, name), ''])
    jsonInfo = {
        'mode': 0,
        'batchSize': [512, 512, 512],
        'data': jsonInfo
    }
    with open(savePath, 'w') as f:
        f.write(json.dumps(jsonInfo))

if __name__ == '__main__':
    GenerateJsonV1()
