import tifffile
import numpy as np
import os
from os.path import join
from scipy.ndimage import zoom

def MaskToSwc():
    # imgPath = r'E:\NeronSegDataSet\HaiNanDataSet1\OriDataSet\mask\43_23_12.tif'
    # swcPath = r'F:\viif01-config\SelectTifLs001\TestResMask\GMM\43_23_12_mask.swc'
    imgPath = r'E:\NeronSegDataSet\HaiNanDataSet1\OriDataSet\mask_5\43_19_8.tif'
    swcPath = r'E:\NeronSegDataSet\HaiNanDataSet1\OriDataSet\43_19_8.swc'
    img = tifffile.imread(imgPath)
    p3d = np.array(np.where(img > 200)).T
    with open(swcPath, 'w') as f:
        count = 1
        for p in p3d:
            if count == 1:
                f.write('%d 0 %.4f %.4f %.4f 1 -1\n' % (count, p[2], p[1], p[0]))
            else:
                f.write('%d 0 %.4f %.4f %.4f 1 %d\n' % (count, p[2], p[1], p[0], count - 1))
            count += 1

def ImgResize():
    path = r'F:\viif01-config\SelectTifLs001\TestResMask\result\48_19_11.tif'
    savePath = r'F:\viif01-config\SelectTifLs001\TestResMask\result\48_19_11_Scale.tif'
    scaleSize = [2, 2, 4]
    img = tifffile.imread(path)
    img = zoom(img, scaleSize[::-1])
    # img.resize(scaleSize[::-1] * img.shape, refcheck=False)
    tifffile.imwrite(savePath, img, compression='lzw')

if __name__ == '__main__':
    MaskToSwc()
    # ImgResize()
