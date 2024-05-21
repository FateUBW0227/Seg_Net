import os, importlib
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from models.model import LoadModel
from torch.utils.data import DataLoader
from DataLoader import GetMultiTypeMemoryDataSetAndCropQxz
import numpy as np
import torch, os
from tensorboardX import SummaryWriter
from Net import Trainer
from MyUtil import GetLossOptimiLr

'''
cmd
activate QxzDeep
cd logs
tensorboard --logdir "./" --host=0.0.0.0


'''

def Train():
    rootPath = 'E:/Training/Paper/Data/training_data' # data path
    trainTxt = "train.txt"         # txt file for training
    valTxt = "test.txt"             # txt file for validation
    imgSize = np.array([128, 128, 128], dtype=np.int32)         # Img size
    batchSize = 2
    device = torch.device('cuda:0')
    logPath = './logs/'             # log dir
    if not os.path.isdir(logPath): os.makedirs(logPath)
    logName = len(os.listdir(logPath))
    expName = 'exp%s' % str(logName).zfill(3)
    logAdd = './logs/' + expName
    while True:
        if os.path.isdir(logAdd):
            logName += 1
            logAdd = './logs/exp%s' % str(logName).zfill(3)
        else:
            break
    writer = SummaryWriter(logAdd)
    savePath = r'./ModelSave/%s' % expName          # path for saving model.
    # load data.
    train_dataset = GetMultiTypeMemoryDataSetAndCropQxz(rootPath, trainTxt, imgSize)
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=0)
    val_dataset = GetMultiTypeMemoryDataSetAndCropQxz(rootPath, valTxt, imgSize)
    val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=True, num_workers=0)
    if not os.path.isdir(savePath): os.makedirs(savePath)
    # load net.
    modelCfg = {
        'name': 'UNet3D',
        # number of input channels to the model
        'in_channels': 16,
        # number of output channels
        'out_channels': 1,
        # determines the order of operators in a single layer (gcr - GroupNorm+Conv3d+ReLU)
        'layer_order': 'gcr',
        # number of features at each level of the U-Net
        # 'f_maps_1': [8, 16],
        # 'f_maps_2': [16, 32, 64, 128],
        # 'addMapsId': 1,
        'f_maps': [16, 32, 64, 128, 256],
        # number of groups in the groupnorm
        'num_groups': 8,
        # apply element-wise nn.Sigmoid after the final 1x1 convolution, otherwise apply nn.Softmax
        # this is only relevant during inference, during training the network outputs logits and it is up to the loss function
        # to normalize with Sigmoid or Softmax
        'final_sigmoid': True,
        # if True applies the final normalization layer (sigmoid or softmax), otherwise the networks returns the output from the final convolution layer; use False for regression problems, e.g. de-noising
        'is_segmentation': True
    }
    # model = LoadModel(modelCfg, r'./ModelSave/exp006/supernet_00000.pth')
    model = LoadModel(modelCfg)
    model.to(device)
    model.train(True)
    loss_criterion, optimizer, lr_scheduler, eval_metric = GetLossOptimiLr(model)
    eval_metric.to(device)

    netObj = Trainer(train_loader, val_loader, model, loss_criterion, optimizer, lr_scheduler, eval_metric, modelPath=savePath, device=device, batchSize=batchSize)
    netObj.Train(turn=300, writer=writer)

if __name__ == '__main__':
    Train()
