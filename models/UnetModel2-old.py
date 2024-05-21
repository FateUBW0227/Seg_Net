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

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, is3d=True):
        super(AbstractUNet, self).__init__()
        self.fieldSpace = 2
        self.in_channels = in_channels
        self.input2 = None

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size, is3d)

        self.encoders2 = create_encoders(1, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size, is3d)

        # create decoder path
        # f_maps_2 = [f * 2 for f in f_maps]
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        is3d)
        self.addDec = nn.ModuleList()
        for f in f_maps[-1::-1]:
            self.addDec.append(DoubleConv(f * 2, f, encoder=False, kernel_size=conv_kernel_size, order=layer_order,
                                         num_groups=num_groups, padding=1, is3d=is3d))
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
        # self.final_activation = None

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
        x1 = self.input2
        encoders_features = []
        for encoder in self.encoders:
            x1 = encoder(x1)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x1)
        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # encoder part2
        x2 = x
        encoders_features2 = []
        for encoder in self.encoders2:
            x2 = encoder(x2)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features2.insert(0, x2)
        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features2 = encoders_features2[1:]

        # decoder part
        x = torch.cat([x1, x2], dim=1)
        x = self.addDec[0](x)
        # x = x1
        for decoder, encoder_features1, encoder_features2, addDec in zip(self.decoders, encoders_features, encoders_features2, self.addDec[1:]):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            # encoder_features = encoder_features1 + encoder_features2
            # encoder_features = encoder_features2
            encoder_features = torch.cat([encoder_features1, encoder_features2], dim=1)
            encoder_features = addDec(encoder_features)
            # encoder_features = encoder_features1
            x = decoder(encoder_features, x)
        x1 = self.final_conv(x)
        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction.
        # During training the network outputs logits
        if not self.training and self.final_activation is not None:
            x1 = self.final_activation(x1)
        return x1

class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
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

def Debug():
    class ConineSimilarity(nn.Module):
        def forward(self, tensor_1, tensor_2):
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)

    ps = np.array([
        [150, 16, 123],
        [157, 21, 112],
        [165, 19, 97],
        [173, 25, 94]
    ], dtype=np.int32)
    yy = encoder_features2
    ps = np.floor(ps / (256 / yy.shape[2])).astype(np.int32)
    ty = yy[..., ps[:, 0], ps[:, 1], ps[:, 2]][0]
    cs = ConineSimilarity()
    print()
    print(cs(ty[:, 0], ty[:, 1]))
    print(cs(ty[:, 0], ty[:, 2]))
    print(cs(ty[:, 0], ty[:, 3]))
    print(cs(ty[:, 1], ty[:, 2]))
    print(cs(ty[:, 1], ty[:, 3]))


    from matplotlib import pyplot as plt
    ps = np.array([
        [150, 16, 123],
        [157, 21, 112],
        [165, 19, 97],
        [173, 25, 94]
    ], dtype=np.int32)
    ty = encoders_features2[-1][..., ps[:, 0], ps[:, 1], ps[:, 2]].detach().cpu().numpy()[0]
    plt.plot(ty[:, 0], c='r')
    plt.plot(ty[:, 1], c='g')
    plt.plot(ty[:, 2], c='b')
    plt.plot(ty[:, 3], c='k')
    plt.show()

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
    obj1 = UNet3D(**modelCfg)
    images = torch.zeros([1, 1, 256, 128, 128], dtype=torch.float32)
    obj1(images)
    # obj2 = ResidualUNet3D(1, 1)
    # obj3 = ResidualUNetSE3D(1, 1)
    # torch.save(obj1.state_dict(), 'tmp4.pth')
    # s1 = obj1.state_dict()
    # s2 = obj2.state_dict()
    # s3 = obj3.state_dict()
    # print()
