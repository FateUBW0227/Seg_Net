'''大数据纤维工具脚本'''

import os, json
import shutil
from os.path import join
import numpy as np

'''基础函数'''

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

# '''Swc多树拆分每条分支
def SplitSwcToBranch(swcData):
    swcDataLs = SplitSwcData(swcData)
    swcBranchDataLs = []
    for swcData in swcDataLs:
        indLs = np.where(swcData[:, 0] - swcData[:, -1] != 1)[0]
        for i in range(indLs.shape[0]):
            sp = indLs[i]
            if i == indLs.shape[0] - 1:
                ep = swcData.shape[0]
            else:
                ep = indLs[i + 1]
            item = swcData[sp: ep]
            tp = swcData[sp, 0] - 1
            item[:, 0] -= tp
            item[1:, -1] -= tp
            item[0, -1] = -1
            swcBranchDataLs.append(item)
    return swcBranchDataLs

# '''保存Big Swc的配置文件Cfg
def SaveSwcCfg(savePath, swcBigCfg):
    swcBigCfg['swcBatchSize1'] = swcBigCfg['swcBatchSize1'].tolist()
    swcBigCfg['swcBatchSize2'] = swcBigCfg['swcBatchSize2'].tolist()
    with open(savePath, 'w') as f: f.write(json.dumps(swcBigCfg))

# '''加载Big Swc的配置文件Cfg
def LoadSwcCfg(cfgPath):
    with open(cfgPath, 'r') as f:
        swcCfg = json.loads(f.read())
        swcCfg['swcBatchSize'] = np.array(swcCfg['swcBatchSize1'], dtype=np.int32)
        swcCfg['swcBatchSize2'] = np.array(swcCfg['swcBatchSize2'], dtype=np.int32)
        return swcCfg

# '''Swc异常数据过滤并转为全局Swc'''
def SwcPreDataToGlobal():
    path = r'Z:\SelectTifLs002\ResSwc\Skeleton'
    savePath = r'D:\qxz\MyProject\KKMarkCellBody\Code\2023_5_18_smallEdit\DataSet\GlobalSwc'
    batchSize = np.array([512, 512, 512], dtype=np.int32)
    redun = 25
    if not os.path.isdir(savePath): os.makedirs(savePath)
    ls = os.listdir(path)
    for name in ls:
        if os.path.splitext(name)[-1] == '.swc':
            swcData = np.loadtxt(join(path, name), ndmin=2, dtype=np.float32)
            swcData[:, 2: 5][swcData[:, 2: 5] < 0] = 0
            swcData[swcData[:, 2] > batchSize[0] - 1, 2] = batchSize[0] - 1
            swcData[swcData[:, 3] > batchSize[1] - 1, 3] = batchSize[1] - 1
            swcData[swcData[:, 4] > batchSize[2] - 1, 4] = batchSize[2] - 1
            name = name.split('new')[0].split('_')
            x, y, z = int(name[0]), int(name[1]), int(name[2])
            sp = [x, y, z] * (batchSize - redun)
            swcData[:, 2: 5] += sp
            np.savetxt(join(savePath, '%d_%d_%d.swc' % (x, y, z)), swcData, fmt='%.8f')

# '''Swc更换根节点'''
def ChangeRootSwc(swcData, ind):
    mark = np.zeros(swcData.shape[0], dtype=np.int32) - 1
    newSwcData = []
    mark[ind] = 0
    count = 1
    curInd = ind
    while True:
        newSwcData.append([count, *swcData[curInd, 1: 6], count - 1])
        mark[curInd] = count - 1
        count += 1
        if swcData[curInd, -1] == -1:
            break
        curInd = int(swcData[curInd, -1]) - 1
    for ii, item in enumerate(swcData):
        if mark[ii] < 0:
            fatherId = int(swcData[ii, -1]) - 1
            newSwcData.append([count, *swcData[ii, 1: 6], mark[fatherId] + 1])
            mark[ii] = count - 1
            count += 1
    newSwcData = np.array(newSwcData, dtype=np.float32)
    newSwcData[0, -1] = -1
    return newSwcData

# '''Swc合并'''
def MergeSwc(swcData1, ind1, swcData2, ind2):
    mark = np.zeros(swcData2.shape[0], dtype=np.int32) - 1
    newSwcData = []
    mark[ind2] = swcData1.shape[0]
    count = swcData1.shape[0] + 1
    curInd = ind2
    while True:
        newSwcData.append([count, *swcData2[curInd, 1: 6], count - 1])
        mark[curInd] = count - 1
        count += 1
        if swcData2[curInd, -1] == -1:
            break
        curInd = int(swcData2[curInd, -1]) - 1
    for ii, item in enumerate(swcData2):
        if mark[ii] < 0:
            fatherId = int(swcData2[ii, -1]) - 1
            newSwcData.append([count, *swcData2[ii, 1: 6], mark[fatherId] + 1])
            mark[ii] = count - 1
            count += 1
    newSwcData[0][-1] = swcData1[ind1, 0]
    resData = np.r_[swcData1, np.asarray(newSwcData)]
    return resData

# '''Swc裁剪'''
def CropSwcData(swcData, ind):
    '''
    Swc裁剪，以裁剪点为新的根节点像下为新树，原树到裁剪点截止。裁剪点不可为根结点，为根结点时数据结果不变
    :param swcData: 单个swc树，nx7
    :param ind: 裁剪点在swcData中的行号
    :return:
    '''
    if swcData[ind, -1] == -1: return False, []
    newSwcDataLs = [[], []]
    markInfo = np.zeros([swcData.shape[0], 2], dtype=np.int32)
    for ii, item in enumerate(swcData):
        curId = int(item[-1])
        if curId == -1:
            newSwcDataLs[0].append(item.tolist())
            markInfo[ii] = [0, 1]
        elif ii == ind:
            newSwcDataLs[1].append([1, item[1], item[2], item[3], item[4], item[5], -1])
            markInfo[ii] = [1, 1]
        else:
            t, fId = markInfo[curId - 1]
            newSwcDataLs[t].append([len(newSwcDataLs[t]) + 1, item[1], item[2], item[3], item[4], item[5], fId])
            markInfo[ii] = [t, len(newSwcDataLs[t])]
    newSwcDataLs = [np.array(item, dtype=np.float32) for item in newSwcDataLs]
    return newSwcDataLs

# '''获取BigSwc名称'''
def GetBigSwcFileName(swcCfg, swcId):
    return join(str(int(swcId / swcCfg['swcFileNumber'])).zfill(swcCfg['swcDirLen']),
                str(swcId).zfill(swcCfg['swcFileLen']) + '.swc')

# '''BigSwc删除'''
def RemoveBigSwc(rootPath, swcCfg, swcId, swcFileName=None, swcIndex=None):
    if swcFileName is None: swcFileName = GetBigSwcFileName(swcCfg, swcId)
    if swcIndex is None:
        with open(join(rootPath, swcCfg['swcDataName'], swcFileName), 'rb') as f: t = f.read()
        swcIndex = np.fromstring(t.split(b'&&')[0], dtype=np.int32).reshape([-1, 3])
    # 删除 SwcData File
    os.remove(join(rootPath, swcCfg['swcDataName'], swcFileName))
    # 更新 SwcIndex File
    for item in swcIndex:
        sp = item * swcCfg['swcBatchSize1']
        sId1 = sp // swcCfg['swcBatchSize2']
        sId2 = (sp - sId1 * swcCfg['swcBatchSize2']) // swcCfg['swcBatchSize1']
        # Update SwcIndex File
        indexAdd = join(rootPath, swcCfg['swcIndexName'], '%d_%d_%d' % (sId1[0], sId1[1], sId1[2]),
                        '%d_%d_%d' % (sId2[0], sId2[1], sId2[2]) + '.txt')
        with open(indexAdd, 'r') as f:
            info = f.read()
            info = info.split(' %d' % swcId)
            info = info[0] + info[1]
        with open(indexAdd, 'w') as f: f.write(info)

# '''BigSwc 读取'''
def ReadBigSwc(swcCfg, swcPath, swcId):
    swcName = GetBigSwcFileName(swcCfg, swcId)
    f = open(join(swcPath, swcName), 'rb')
    t = f.read()
    f.close()
    tt = t.split(b'&&')
    swcIndex = np.fromstring(tt[0], dtype=np.int32).reshape([-1, 3])
    swcData = np.fromstring(tt[1], dtype=np.float32).reshape([-1, 7])
    return swcIndex, swcData, swcName

# '''添加大数据Swc Ls'''
def AddBigSwcLs(rootPath, swcBigCfg, swcDataLs):
    # 确定新的SwcId
    dataPath = join(rootPath, swcBigCfg['swcDataName'])
    maxDir = os.listdir(dataPath)[-1]
    swcId = int(os.path.splitext(os.listdir(join(dataPath, maxDir))[-1])[0]) + 1
    curSwcDirAdd = join(dataPath, str(int(swcId / swcBigCfg['swcFileNumber'])).zfill(swcBigCfg['swcDirLen']))
    if not os.path.isdir(curSwcDirAdd): os.makedirs(curSwcDirAdd)
    swcIndexDict = {}
    resSwcData = {}
    # 保存每一个SwcData
    for data in swcDataLs:
        data = data.astype(np.float32)
        if swcId % swcBigCfg['swcFileNumber'] == 0:
            curSwcDirAdd = join(dataPath, str(int(swcId / swcBigCfg['swcFileNumber'])).zfill(swcBigCfg['swcDirLen']))
            if not os.path.isdir(curSwcDirAdd): os.makedirs(curSwcDirAdd)
        batchIdInfo = np.unique((data[:, 2: 5] // swcBigCfg['swcBatchSize1']).astype(np.int32),
                                axis=0)  # swc到batchId索引信息
        with open(join(curSwcDirAdd, str(swcId).zfill(swcBigCfg['swcFileLen']) + '.swc'), 'wb') as f:
            f.write(batchIdInfo.tostring() + '&&'.encode() + data.tostring())
        # 记录Swc Index信息
        for batchId in batchIdInfo:
            iId = '%d_%d_%d' % (batchId[0], batchId[1], batchId[2])
            if iId in swcIndexDict:
                swcIndexDict[iId] += ' %d' % swcId
            else:
                swcIndexDict[iId] = ' %d' % swcId
        resSwcData[swcId] = data
        swcId += 1
    # 保存SwcIndex信息
    swcIndexPath = join(rootPath, swcBigCfg['swcIndexName'])
    for key in swcIndexDict:
        sp = np.array(key.split('_'), dtype=np.int32) * swcBigCfg['swcBatchSize1']
        sId1 = sp // swcBigCfg['swcBatchSize2']
        sId2 = (sp - sId1 * swcBigCfg['swcBatchSize2']) // swcBigCfg['swcBatchSize1']
        indexAdd = join(swcIndexPath, '%d_%d_%d' % (sId1[0], sId1[1], sId1[2]))
        indexAdd2 = join(indexAdd, '%d_%d_%d' % (sId2[0], sId2[1], sId2[2]) + '.txt')
        # 更新 SwcIndex
        text = swcIndexDict[key]
        if os.path.isfile(indexAdd2):
            with open(indexAdd2, 'r') as f:
                text2 = f.read()
            text = text2 + text
        else:
            os.makedirs(indexAdd, exist_ok=True)
        with open(indexAdd2, 'w') as f:
            f.write(text)
    return resSwcData

# '''Swc拆分插值'''
def SwcSplitAndInter(swcData, interThre):
    mark = np.zeros(swcData.shape[0], dtype=np.int32)
    newSwcData = []
    count = 1
    for i in range(swcData.shape[0]):
        if swcData[i, -1] == -1:
            newSwcData.append([swcData[i].tolist()])
            newSwcData[-1][0][0] = 1
            count = 2
            mark[i] = 0
        else:
            data2 = newSwcData[-1]
            fId = int(swcData[i, -1] - 1)
            dp = swcData[i, 2: 5] - swcData[fId, 2: 5]
            interNum = np.abs(dp).max() / interThre
            if interNum >= 1:
                interNum = int(np.ceil(interNum)) + 1
                pp = swcData[fId, 2: 5] + dp * 1 / interNum
                data2.append([count, swcData[fId, 1], *pp, swcData[fId, 5], mark[fId] + 1])
                count += 1
                for t in range(2, interNum):
                    pp = swcData[fId, 2: 5] + dp * t / interNum
                    data2.append([count, swcData[fId, 1], *pp, swcData[fId, 5], len(data2)])
                    count += 1
                data2.append([count, *swcData[i, 1: 6], len(data2)])
                count += 1
            else:
                data2.append([count, *swcData[i, 1: 6], mark[fId] + 1])
                count += 1
            mark[i] = len(data2) - 1
    return newSwcData

# '''Swc插值得到得到点云'''
def SwcInterToPc(swcData, interThre=1):
    newSwcData = []
    for i in range(swcData.shape[0]):
        if swcData[i, -1] == -1:
            newSwcData.append([swcData[i, 2], swcData[i, 3], swcData[i, 4]])
        else:
            p0 = swcData[i, 2: 5]
            p1 = swcData[int(swcData[i, -1] - 1), 2: 5]
            d = np.linalg.norm(p0 - p1)
            if d > interThre:
                dir = (p1 - p0) / d
                x = np.append(np.arange(0, d, interThre / 2), d).reshape([-1, 1])
                newP = p0 + dir * x
                newSwcData += newP.tolist()
            else:
                newSwcData.append([swcData[i, 2], swcData[i, 3], swcData[i, 4]])
    return np.array(newSwcData)

# '''保存点云列表到Swc中
def SavePCLsToSwc(savePath, pcLs):
    with open(savePath, 'w') as f:
        count = 1
        for pc in pcLs:
            f.write('%d 1 %.4f %.4f %.4f 4 -1\n' % (count, pc[0, 0], pc[0, 1], pc[0, 2]))
            count += 1
            for p in pc[1:]:
                f.write('%d 1 %.4f %.4f %.4f 1 %d\n' % (count, p[0], p[1], p[2], count - 1))
                count += 1

# '''大数据坐标转块Id和swc
def BigCoorToBatchIdAndSwc(coor, savePath, batchSize=np.array([512, 512, 512]), redun=25):
    batchSize2 = batchSize - redun
    batchId = (coor / batchSize2).astype(np.int32)
    sp = batchId * batchSize2
    p = coor - sp
    with open(join(savePath, '%d_%d_%d_ttt.swc' % (batchId[0], batchId[1], batchId[2])), 'w') as f:
        f.write('1 1 %d %d %d 1 -1\n' % (p[0], p[1], p[2]))
        f.write('2 1 %d %d %d 1 1' % (511, 511, 511))
    print(batchId)

'''功能函数'''
# Swc点云保存
def SavePcToSwc(savePath, data):
    with open(savePath, 'w') as f:
        count = 1
        for ii, item in enumerate(data):
            if ii == 0:
                f.write('%d %d %.8f %.8f %.8f %.4f %d\n' % (count, 0, item[0], item[1], item[2], 0, -1))
            else:
                f.write('%d %d %.8f %.8f %.8f %.4f %d\n' % (count, 0, item[0], item[1], item[2], 0, count - 1))
            count += 1

# 多树Swc拆分并保存
def SplitSwcDataSaveFun():
    path = r'D:\qxz\MyProject\KKMarkCellBody\Code\2023_5_18_smallEdit\DataSet\GlobalSwc\47_21_11.swc'
    savePath = r'D:\qxz\MyProject\KKMarkCellBody\Code\2023_5_18_smallEdit\DataSet\SignSwc'
    if not os.path.isdir(savePath): os.makedirs(savePath)
    swcData = np.loadtxt(path, ndmin=2, dtype=np.float32)
    swcDataLs = SplitSwcData(swcData)
    for ii, data in enumerate(swcDataLs):
        np.savetxt(join(savePath, str(ii).zfill(5) + '.swc'), data, fmt='%.8f')

# Swc裁剪并保存
def CropSwcSaveFun():
    path = r'D:\qxz\MyProject\KKMarkCellBody\Code\2023_5_18_smallEdit\DataSet\SignSwc\00008.swc'
    savePath = r'D:\qxz\MyProject\KKMarkCellBody\Code\2023_5_18_smallEdit\DataSet'
    ind = 99
    os.makedirs(savePath, exist_ok=True)
    swcData = np.loadtxt(path, ndmin=2, dtype=np.float32)
    newSwcDataLs = CropSwcData(swcData, ind)
    for ii, data in enumerate(newSwcDataLs):
        np.savetxt(join(savePath, str(ii).zfill(5) + '.swc'), data, fmt='%.8f')

# 更换Swc根节点并保存
def ChangeRootSwcSaveFun():
    swcData = np.loadtxt(r'D:\qxz\MyProject\KKMarkCellBody\Code\2023_5_18_smallEdit\DataSet\SignSwc\00007.swc', ndmin=2,
                         dtype=np.float32)
    newSwcData = ChangeRootSwc(swcData, 21)
    np.savetxt('tmp2.swc', newSwcData, fmt='%.8f')

# 合并Swc并保存
def MergeSwcSaveFun():
    swcData1 = np.loadtxt(r'D:\qxz\MyProject\KKMarkCellBody\Code\2023_5_18_smallEdit\DataSet\SignSwc\00006.swc',
                          ndmin=2,
                          dtype=np.float32)
    ind1 = 21
    swcData2 = np.loadtxt(r'D:\qxz\MyProject\KKMarkCellBody\Code\2023_5_18_smallEdit\DataSet\SignSwc\00007.swc',
                          ndmin=2,
                          dtype=np.float32)
    ind2 = 21
    mergeData = MergeSwc(swcData1, ind1, swcData2, ind2)
    np.savetxt('tmp3.swc', mergeData, fmt='%.8f')

# Swc转到全局Swc
def SwcToGlobalSwcFun():
    path = r'Z:\SelectTifLs003\ResSwc\Skeleton'
    savePath = r'Z:\SelectTifLs003\ResSwcGlobal'
    redun = 25
    batchSize = np.array([512, 512, 512], dtype=np.int32)
    if not os.path.isdir(savePath): os.makedirs(savePath)
    ls = os.listdir(path)
    for name in ls:
        add = join(path, name)
        if os.path.isdir(add): continue
        nameId, nameType = os.path.splitext(name)
        if nameType == '.swc':
            tmp = nameId.split('_')
            x, y, z = int(tmp[0]), int(tmp[1]), int(tmp[2])
            sp = [x, y, z] * (batchSize - redun)
            data = np.loadtxt(add, ndmin=2, dtype=np.float32)
            data[:, 2: 5] += sp
            np.savetxt(join(savePath, '%d_%d_%d.swc' % (x, y, z)), data, fmt='%.8f')

# '''Swc放缩代码'''
def SwcScaleFun():
    path = r'D:\qxz\MyProject\KKMarkCellBody\Code\2023_5_18_smallEdit\DataSet\DataSwc'
    savePath = r'D:\qxz\MyProject\KKMarkCellBody\Code\2023_5_18_smallEdit\DataSet\DataSwcLevel0'
    if not os.path.isdir(savePath): os.makedirs(savePath)
    ls = os.listdir(path)
    for name in ls:
        if os.path.splitext(name)[-1] == '.swc':
            data = np.loadtxt(join(path, name))
            data[:, 2: 5] *= 2
            # np.savetxt(join(savePath, name), data, fmt='%.8f')
            with open(join(savePath, name), 'w') as f:
                for item in data:
                    f.write('%d %d %f %f %f %d %d\n' % (item[0], item[1], item[2], item[3], item[4], item[5], item[6]))
            #     count = 0
            #     for item in data:
            #         if count == 0:
            #             f.write('%d %d %f %f %f %f -1\n' % (item[0], item[1], item[2], item[3], item[4], item[5]))
            #         else:
            #             f.write('%d %d %f %f %f %f %d\n' % (item[0], item[1], item[2], item[3], item[4], item[5], count))
            #         count += 1

# '''Swc拆分并插值'''
def SwcSplitAndInterFun():
    path = r'D:\qxz\MyProject\KKMarkCellBody\Code\2023_5_18_smallEdit\DataSet\GlobalScaleSwc'
    savePath = r'D:\qxz\MyProject\KKMarkCellBody\Code\2023_5_18_smallEdit\DataSet\TestSwc'
    name = '47_21_11.swc'
    os.makedirs(savePath, exist_ok=True)
    swcData = np.loadtxt(join(path, name), ndmin=2, dtype=np.float32)
    newSwcLs = SwcSplitAndInter(swcData, 0.2)
    for ii, swcData in enumerate(newSwcLs):
        np.savetxt(join(savePath, str(ii).zfill(3) + '.swc'), swcData, fmt='%.8f')

# '''Swc多树保存到一个Swc中'''
def SaveSwcLs(savePath, threeDataLs):
    sp = 0
    newSwcData = []
    for ii, item in enumerate(threeDataLs):
        item = np.array(item)
        print(ii, sp)
        item[:, 0] += sp
        item[1:, -1] += sp
        newSwcData += item.tolist()
        sp = item[-1, 0]
    with open(savePath, 'w') as f:
        for item in newSwcData:
            f.write('%d %d %.8f %.8f %.8f %.8f %d\n' % (item[0], item[1], item[2], item[3], item[4], item[5], item[6]))
    # np.savetxt(savePath, newSwcData, fmt='%.8f')

# '''Swc树插值保存'''
def SwcInterSave():
    path = r'D:\qxz\MyProject\KKMarkCellBody\Code\data\VIIF01重建23823\VIIO1重建23823'
    savePath = r'D:\qxz\MyProject\KKMarkCellBody\Code\data\VIIF01重建23823\VIIO1重建23823_New'
    dirLs = os.listdir(path)
    for dir in dirLs:
        add = join(path, dir)
        ls = os.listdir(add)
        for name in ls:
            saveAdd = join(savePath, dir)
            os.makedirs(saveAdd, exist_ok=True)
            if name.split('-')[1].split('.')[0] in ['axon', 'den']:
                swcData = np.loadtxt(join(add, name), ndmin=2)
                newSwcData = SwcSplitAndInter(swcData, 0.1)
                # newSwcData = np.array(newSwcData)
                SaveSwcLs(join(saveAdd, name), newSwcData)
                # newSwcData = SwcSplitAndInter(swcData, 1)[0]
                # with open(join(saveAdd, name), 'w') as f:
                #     for item in newSwcData:
                #         f.write('%d %d %.8f %.8f %.8f %.8f %d\n' % (item[0], item[1], item[2], item[3], item[4], item[5], item[6]))
            else:
                swcData = np.loadtxt(join(add, name), ndmin=2)
                with open(join(saveAdd, name), 'w') as f:
                    for item in swcData:
                        f.write('%d %d %.8f %.8f %.8f %.8f %d\n' % (item[0], item[1], item[2], item[3], item[4], item[5], item[6]))

# ‘’‘批量分辨率修改'''
def SwcBatchResulutAlter():
    path = r''
    savePath = r''
    resulut = np.array([0.32, 0.32, 1], dtype=np.float32)
    os.makedirs(savePath, exist_ok=True)
    ls = os.listdir(path)
    for name in ls:
        swcData = np.loadtxt(join(path, name), ndmin=2)
        swcData[:, 2: 5] = swcData[:, 2: 5] * resulut
        with open(join(savePath, name), 'w') as f:
            for item in swcData:
                f.write(
                    '%d %d %.8f %.8f %.8f %.8f %d\n' % (item[0], item[1], item[2], item[3], item[4], item[5], item[6]))

# ’‘’大数据坐标转Swc和块Id函数
def BigCoorToBatchIdAndSwcFun():
    coor = np.array([10511, 13985, 2306])
    savePath = r'D:\qxz\MyProject\KKMarkCellBody\Code\NeuronTrack\DataSet\ProblemDataSet'
    BigCoorToBatchIdAndSwc(coor, savePath)

if __name__ == '__main__':
    SavePCLsToSwc()
    # SplitSwcDataSaveFun()
    # CropSwcSaveFun()
    # SwcScale()
    # ChangeRootSwcSaveFun()
    # MergeSwcSaveFun()
    # SwcToGlobalSwcFun()
    # SwcScaleFun()
    # SwcSplitAndInterFun()
    # SwcInterSave()
    # SwcBatchResulutAlter()
    BigCoorToBatchIdAndSwcFun()
