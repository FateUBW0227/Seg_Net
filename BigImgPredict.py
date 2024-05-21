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
# from skimage import exposure

if __name__ == '__main__':
 #   imgPath = r'/media/MD3400-2/Cailin/test_data'
 #   savePath = r'/media/MD3400-2/Cailin/training_data/predict_result_exp025_64pth'
    imgPath = r'./test_data'
    savePath = r'./save_data'

    modelPath = r'./ModelSave/exp025/supernet_00030.pth'
    bigImgSize = np.array([512, 512, 512], dtype=np.int32)
    imgSize = np.array([128, 128, 128], dtype=np.int32)
    r = 32          # 冗余
    rSp = 16        # 边缘不要部分
    batchSize = 1
    fieldLen = 16
    device = torch.device('cuda:1')

    os.makedirs(savePath, exist_ok=True)
    model = ModelPredictClass(modelPath, fieldLen=fieldLen, device=device)  # 预测类
    # # ls = [os.listdir(imgPath)[220]]
    # ls = ['10623_8104_831.tif']
    ls = os.listdir(imgPath)
    saveInfo = []
    for ii, name in enumerate(ls):
        s1 = time.time()
        oriImg = tifffile.imread(join(imgPath, name))
        # oriImg = (oriImg - oriImg.min()) / (oriImg.max() - oriImg.min())
        # oriImg = (exposure.equalize_hist(oriImg) * 2700).astype(np.uint16)
        # tifffile.imwrite(r'D:\qxz\MyProject\KKMarkCellBody\Code\NeuronTrack\DataSet\ProblemDataSet\SegProblemTest\SmallImages\ttt.tif', oriImg)
        # oriImg[oriImg > 125] = 125
        # oriImg
        if oriImg.ndim != 3: continue
        maskBigImg = np.zeros(bigImgSize[::-1], dtype=np.uint8)
        sliceNumber = np.ceil((bigImgSize - imgSize) / (imgSize - r)).astype(np.int32) + 1
        newName = os.path.splitext(name)[0]
        for nz in range(sliceNumber[2]):
            for ny in range(sliceNumber[1]):
                for nx in range(sliceNumber[0]):
                    sp = (imgSize - r) * [nx, ny, nz]
                    ep = np.min([sp + imgSize, bigImgSize], axis=0)
                    sp = np.min([sp, ep - imgSize], axis=0)
                    img = oriImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]]
                    # stdVal = img.std()
                    # if stdVal > 500:
                    #     print('**********************')
                    # print(name, nx, ny, nz, img.std())
                    mask = model(img)
                    rsp2 = rSp * np.sign([nx, ny, nz])
                    sp += rsp2
                    maskBigImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]] = mask[rsp2[2]:, rsp2[1]:, rsp2[0]:]
        maskBigImg[maskBigImg < 103] = 0
        tifffile.imwrite(join(savePath, '%s.tif' % newName), maskBigImg)
        # saveInfo.append([join(saveRes, '%s.tif' % newName), join(imgPath, name), ''])
        # with open(cfgPath, 'w') as f:
        #     f.write(json.dumps(saveInfo))
        print(ii, len(ls), time.time() - s1)
