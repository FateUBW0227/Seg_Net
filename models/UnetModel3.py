import numpy as np
import torch.nn as nn
from pytorch3dunet.unet3d.buildingblocks import DoubleConv, ResNetBlock, ResNetBlockSE, \
    create_decoders, create_encoders
from pytorch3dunet.unet3d.utils import get_class, number_of_features_per_level
import torch

class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution,
            otherwise apply nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True and the model is in eval mode, Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): if True the model is 3D, otherwise 2D, default: True
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps_1=[64], f_maps_2=[64], addMapsId=0, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, is3d=True):
        super(AbstractUNet, self).__init__()
        self.fieldSpace = 2
        self.in_channels = in_channels
        self.input2 = None
        f_maps = sorted(list(set(f_maps_1 + f_maps_2)))[1:]
        self.addMapsId = addMapsId
        self.AddConv = DoubleConv(f_maps_1[-1] + f_maps_2[addMapsId], f_maps_2[addMapsId], encoder=True, kernel_size=conv_kernel_size, order=layer_order,
                                         num_groups=num_groups, padding=1, is3d=is3d)
        # create encoder path
        self.encoders1 = create_encoders(1, f_maps_1, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size, is3d)
        self.encoders2 = create_encoders(in_channels, f_maps_2, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size, is3d)
        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        is3d)
        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        if is3d:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)
        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        if self.input2 is None:
            self.batchSize = x.size(0)
            self.imgSize = x.shape[-3:][::-1]
            self.device = x.device
            self.centZ = self.imgSize[2] // 2
            self.centY = self.imgSize[1] // 2
            self.centX = self.imgSize[0] // 2
            self.fieldLen = (self.in_channels - 1) // 3
            self.input2 = torch.zeros([self.batchSize, self.in_channels, *self.imgSize[::-1]], dtype=torch.float32).to(
                self.device)
        self.input2[:, 0] = x[:, 0]
        for i in range(1, self.fieldLen + 1):
            j = i * self.fieldSpace
            # x轴
            self.input2[:, i, :, :, :self.centX] = x[:, 0, :, :, j: self.centX + j]
            self.input2[:, i, :, :, self.centX:] = x[:, 0, :, :, self.centX - j: self.imgSize[0] - j]
            # y轴
            self.input2[:, i + self.fieldLen, :, :self.centY] = x[:, 0, :, j: self.centY + j]
            self.input2[:, i + self.fieldLen, :, self.centY:] = x[:, 0, :, self.centY - j: self.imgSize[1] - j]
            # z轴
            self.input2[:, i + self.fieldLen * 2, :self.centZ] = x[:, 0, j: self.centZ + j]
            self.input2[:, i + self.fieldLen * 2, self.centZ:] = x[:, 0, self.centZ - j: self.imgSize[2] - j]
        self.input2 = self.input2 / (x[:, [0]] + 0.01)
        # encoder part
        x1 = x
        for encoder in self.encoders1:
            x1 = encoder(x1)
        encoders_features = []
        x2 = self.input2
        for ei, encoder in enumerate(self.encoders2):
            x2 = encoder(x2)
            if ei == self.addMapsId:
                x2 = torch.cat([x2, x1], dim=1)
                x2 = self.AddConv(x2)
            encoders_features.insert(0, x2)
        encoders_features = encoders_features[1:]
        # decoder part
        for decoder, encoders_feature in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x2 = decoder(encoders_feature, x2)
        x2 = self.final_conv(x2)
        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction.
        # During training the network outputs logits
        if not self.training and self.final_activation is not None:
            x2 = self.final_activation(x2)
        return x2

class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps_1=64, f_maps_2=64, addMapsId=0, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps_1=f_maps_1,
                                     f_maps_2=f_maps_2,
                                     addMapsId=addMapsId,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     is3d=True)

class ResidualUNet3D(AbstractUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             is3d=True)

class ResidualUNetSE3D(AbstractUNet):
    """_summary_
    Residual 3DUnet model implementation with squeeze and excitation based on
    https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlockSE as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch
    out for block artifacts). Since the model effectively becomes a residual
    net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, **kwargs):
        super(ResidualUNetSE3D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               final_sigmoid=final_sigmoid,
                                               basic_module=ResNetBlockSE,
                                               f_maps=f_maps,
                                               layer_order=layer_order,
                                               num_groups=num_groups,
                                               num_levels=num_levels,
                                               is_segmentation=is_segmentation,
                                               conv_padding=conv_padding,
                                               is3d=True)

class UNet2D(AbstractUNet):
    """
    2DUnet model from
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/abs/1505.04597>`
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     is3d=False)

def get_model(model_config):
    model_class = get_class(model_config['name'], modules=[
        'pytorch3dunet.unet3d.model'
    ])
    return model_class(**model_config)

if __name__ == '__main__':
    modelCfg = {
        'name': 'UNet3D',
        # number of input channels to the model
        'in_channels': 16,
        # number of output channels
        'out_channels': 1,
        # determines the order of operators in a single layer (gcr - GroupNorm+Conv3d+ReLU)
        'layer_order': 'gcr',
        # number of features at each level of the U-Net
        'f_maps': [32, 64, 128, 256],
        # number of groups in the groupnorm
        'num_groups': 8,
        # apply element-wise nn.Sigmoid after the final 1x1 convolution, otherwise apply nn.Softmax
        # this is only relevant during inference, during training the network outputs logits and it is up to the loss function
        # to normalize with Sigmoid or Softmax
        'final_sigmoid': True,
        # if True applies the final normalization layer (sigmoid or softmax), otherwise the networks returns the output from the final convolution layer; use False for regression problems, e.g. de-noising
        'is_segmentation': True
    }
    obj1 = UNet3D(**modelCfg)
    # obj2 = ResidualUNet3D(1, 1)
    # obj3 = ResidualUNetSE3D(1, 1)
    # torch.save(obj1.state_dict(), 'tmp4.pth')
    # s1 = obj1.state_dict()
    # s2 = obj2.state_dict()
    # s3 = obj3.state_dict()
    # print()
