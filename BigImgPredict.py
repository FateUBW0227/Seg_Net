'''大图预测'''
import json
import os
import shutil

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time, torch
import numpy as np
import tifffile, os
from os.path import join
from ModelPredictPy import ModelPredictClass

if __name__ == '__main__':
    imgPath = r"D:\qxz\MyProject\KKMarkCellBody\Code\BrainSegNewQxz\SegmentProblem\DataSet\img"
    modelPath = r'./ModelSave/exp006/supernet_00010.pth'
    # testTxt = r'test.txt'
    savePath = r'D:\qxz\MyProject\KKMarkCellBody\Code\BrainSegNewQxz\SegmentProblem\DataSet\mask'
    bigImgSize = np.array([512, 512, 512], dtype=np.int32)
    imgSize = np.array([128, 128, 128], dtype=np.int32)
    batchSize = 1
    fieldLen = 16
    device = torch.device('cuda:0')

    # cfgPath = join(savePath, 'info.json')
    # saveRes = join(savePath, 'result')

    # if os.path.isdir(savePath): shutil.rmtree(savePath)
    # os.makedirs(savePath)
    # if os.path.isdir(saveRes): shutil.rmtree(saveRes)
    # os.makedirs(saveRes)

    saveRes = savePath
    os.makedirs(saveRes, exist_ok=True)

    model = ModelPredictClass(modelPath, fieldLen=fieldLen, device=device)  # 预测类
    # with open(join(rootPath, testTxt), 'r') as f:
    #     ls = f.read().strip().split('\n')
    # imgPath = join(rootPath, 'images')
    ls = os.listdir(imgPath)

    saveInfo = []
    for name in ls:
        oriImg = tifffile.imread(join(imgPath, name))
        if oriImg.ndim != 3: continue
        # oriImg_eq = exposure.equalize_hist(oriImg, nbins=oriImg.max() - oriImg.min())
        # newImg = np.zeros_like(oriImg)
        maskBigImg = np.zeros(bigImgSize[::-1], dtype=np.uint8)
        sliceNumber = bigImgSize // imgSize
        count = 0
        newName = os.path.splitext(name)[0]
        for nz in range(sliceNumber[2]):
            for ny in range(sliceNumber[1]):
                for nx in range(sliceNumber[0]):
                    s1 = time.time()
                    sp = imgSize * [nx, ny, nz]
                    ep = sp + imgSize
                    img = oriImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]]
                    # img_eq = oriImg_eq[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]]
                    # img = np.array([img, img_eq], dtype=np.float32)

                    # newImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]] = img
                    mask = model(img)
                    maskBigImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]] = mask
                    print(count, time.time() - s1)
                    count += 1
                    # tifffile.imwrite(join(savePath, '%s_%s_0.tif' % (newName, str(count).zfill(5))), img)
                    # tifffile.imwrite(join(savePath, '%s_%s_1.tif' % (newName, str(count).zfill(5))), mask)
                    # print('%s_%s_1.tif' % (newName, str(count).zfill(5)), mask.sum())

                    # break
                # break
            # break
        # tifffile.imwrite(join(savePath, '%s_0.tif' % newName), oriImg[:bigImgSize[2], :bigImgSize[1], :bigImgSize[0]])
        maskBigImg[maskBigImg < 103] = 0
        tifffile.imwrite(join(saveRes, '%s.tif' % newName), maskBigImg)
        # saveInfo.append([join(saveRes, '%s.tif' % newName), join(imgPath, name), ''])
        # with open(cfgPath, 'w') as f:
        #     f.write(json.dumps(saveInfo))
