'''数据预处理'''

import os, tifffile
import shutil

import numpy as np
from os.path import join


# '''Swc多树拆分'''
def SplitSwcData(swcData):
    indLs = np.where(swcData[:, -1] == -1)[0].tolist() + [swcData.shape[0]]
    swcDataLs = []
    for i in range(len(indLs) - 1):
        data = swcData[indLs[i]: indLs[i + 1]]
        sp = data[0, 0]
        data[:, 0] -= sp - 1
        data[1:, -1] -= sp - 1
        swcDataLs.append(data)
    return swcDataLs


'''获取点云核点云'''
def GetPcKernelPc(pc, kernelArr, imgShape):
    curPc = (pc[:, None] + kernelArr[None]).reshape([-1, 3])
    curPc = np.round(curPc).astype(np.int32)
    curPc[curPc < 0] = 0
    curPc[curPc[:, 2] > imgShape[0] - 1] = imgShape[0] - 1
    curPc[curPc[:, 1] > imgShape[1] - 1] = imgShape[1] - 1
    curPc[curPc[:, 0] > imgShape[2] - 1] = imgShape[2] - 1
    curPc = np.unique(curPc, axis=0)
    return curPc


'''纤维转距离场'''
def SwcToDF():
    swcPath = r'.\swc_files'
    savePath = r'.\mask'
    kernelLen = 3
    imgShape = np.array([512, 512, 512], dtype=np.int32)
    maxD = ((kernelLen ** 2) * 3) ** 0.5
    minD = 0.0697
    maskMax = -np.log(minD)
    dfImg = np.zeros(imgShape[::-1], dtype=np.float32)
    kernelArr = []
    for z in range(-kernelLen, kernelLen + 1):
        for y in range(-kernelLen, kernelLen + 1):
            for x in range(-kernelLen, kernelLen + 1):
                kernelArr.append([x, y, z])
    kernelArr = np.array(kernelArr, dtype=np.int32)
    os.makedirs(savePath, exist_ok=True)
    ls = os.listdir(swcPath)
    for name in ls:
        nameId = os.path.splitext(name)[0]
        swcData = np.loadtxt(join(swcPath, name), ndmin=2)
        swcDataLs = SplitSwcData(swcData)
        dfImg[...] = maxD
        for swcData in swcDataLs:
            for ii, item in enumerate(swcData):
                if item[-1] == -1: continue
                p0 = item[2: 5]
                p1 = swcData[int(item[-1]) - 1, 2: 5]
                v = (p1 - p0).reshape([1, 3])
                if np.linalg.norm(v) < 0.1: continue
                # 插值
                d = np.linalg.norm(p0 - p1)
                if d > kernelLen:
                    d2 = int(d + 1)
                    xLs = (np.arange(1, d2 + 1, 1) / d2).reshape([-1, 1])
                    data = p0 * xLs + p1 * (1 - xLs)
                else:
                    data = np.array([p0, p1])
                curPc = GetPcKernelPc(data, kernelArr, imgShape)
                proT = (curPc - p0).dot(v.T) / v.dot(v.T)
                proT = np.clip(proT, 0, 1)
                proP = p0 + proT * v
                d = np.linalg.norm(proP - curPc, axis=1)
                if np.isnan(d).any():
                    print()
                dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]] = np.min([dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]], d], axis=0)
        dfImg = -np.log(minD + dfImg / maxD * (1- minD))
        dfImg2 = (dfImg / maskMax * 255).astype(np.uint8)
        # dfImg = ((dfImg - dfImg.min()) / (dfImg.max() - dfImg.min()) * 255).astype(np.uint8)
        # dfImg2[dfImg2 < 103] = 0
        tifffile.imwrite(join(savePath, nameId + '.tif'), dfImg2, compression='lzw')


'''Mask转Swc'''
def MaskToSwc():
    maskPath = r''
    swcPath = r''
    thre = 135
    os.makedirs(swcPath, exist_ok=True)
    ls = os.listdir(maskPath)
    for name in ls:
        img = tifffile.imread(join(maskPath, name))
        pc = np.where(img > thre)
        pc = np.array(pc).T
        with open(join(swcPath, os.path.splitext(name)[0] + '.swc'), 'w') as f:
            for ii, item in enumerate(pc):
                f.write('%d %d %d %d %d %d -1\n' % (ii + 1, 1, item[2], item[1], item[0], 1))


'''Mask改值转Mask'''
def MaskToMask():
    path = r''
    savePath = r''
    os.makedirs(savePath, exist_ok=True)
    ls = os.listdir(path)
    for name in ls:
        img = tifffile.imread(join(path, name))
        img[img < 108] = 0
        tifffile.imwrite(join(savePath, name), img)


def SwcGetTestSwc():
    path = r''
    swcPath = r''
    savePath = r''
    os.makedirs(savePath, exist_ok=True)
    with open(path, 'r') as f:
        dataIdLs = f.read().strip().split('\n')
    for dataId in dataIdLs:
        name = os.path.splitext(dataId)[0] + '.swc'
        shutil.copy(join(swcPath, name), join(savePath, name))


if __name__ == '__main__':
    SwcToDF()
    # MaskToSwc()
    # img = tifffile.imread(r'D:\qxz\MyProject\KKMarkCellBody\Code\NeuronTrack\DataSet\TrainDataSet\mask\11_7_5.tif')
    # print(img.min(), img.max())
    # MaskToMask()
    # SwcGetTestSwc()
