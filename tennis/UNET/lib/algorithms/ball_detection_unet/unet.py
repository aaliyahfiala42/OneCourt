from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from gaussian_smoothing import GaussianSmoothing


GAUSSIAN_SMOOTHING = GaussianSmoothing(channels=1, kernel_size=5, sigma=3, padding=2)
GAUSSIAN_SMOOTHING.weight = GAUSSIAN_SMOOTHING.weight.to(torch.device("cpu"))


def unet(**kwargs):
    """
    U-Net segmentation model with batch normalization for biomedical image segmentation
    pretrained (bool): load pretrained weights into the model
    in_channels (int): number of input channels
    out_channels (int): number of output channels
    init_features (int): number of feature-maps in the first encoder layer
    """
    model = UNet(**kwargs)
    return model

class UNet(nn.Module):

    def __init__(self, in_channels=9, out_channels=1, init_features=16, exporting=False):
        super(UNet, self).__init__()
        self.exporting = exporting

        features = init_features

        self.downsample1 = nn.Conv2d(
                            in_channels=3,
                            out_channels=3,
                            kernel_size=3,
                            padding=1,
                            stride=2,
                            bias=False,
                            groups=3
                        )

        self.downsample_bn1 = nn.BatchNorm2d(num_features=3)
        self.downsample_relu1 = nn.ReLU(inplace=True)

        self.downsample2 = nn.Conv2d(
                            in_channels=3,
                            out_channels=3,
                            kernel_size=3,
                            padding=1,
                            stride=2,
                            bias=False,
                            groups=3
                        )

        self.downsample_bn2 = nn.BatchNorm2d(num_features=3)
        self.downsample_relu2 = nn.ReLU(inplace=True)

        self.downsample3 = nn.Conv2d(
                            in_channels=3,
                            out_channels=3,
                            kernel_size=3,
                            padding=1,
                            stride=2,
                            bias=False,
                            groups=3
                        )

        self.downsample_bn3 = nn.BatchNorm2d(num_features=3)
        self.downsample_relu3 = nn.ReLU(inplace=True)



        self.encoder1 = UNet._block(in_channels, features, name="enc1_mod")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=1, stride=1
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv_mod = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x1, x2, x3):
    # def forward(self, x):
        if self.exporting:
            x = torch.cat((x1, x2, x3), dim=1)


        x1 = self.downsample1(x1)

        x2 = self.downsample2(x2)

        x3 = self.downsample3(x3)


        x1 = self.downsample_bn1(x1)
        x1 = self.downsample_relu1(x1)

        x2 = self.downsample_bn2(x2)
        x2 = self.downsample_relu2(x2)

        x3 = self.downsample_bn3(x3)
        x3 = self.downsample_relu3(x3)

        x = torch.cat((x1, x2, x3), dim=1)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(enc4)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        if self.exporting:
            return self.find_local_max(GAUSSIAN_SMOOTHING(torch.sigmoid(self.conv_mod(dec1))))
        else:
            return torch.sigmoid(self.conv_mod(dec1))


    def find_local_max(self, prediction):
        # shift_left = torch.cat((prediction[:, :, :, 1:], torch.zeros(1, prediction.shape[1], prediction.shape[2], 1).to(prediction.device)), 3) 
        # shift_right = torch.cat((torch.zeros(1, prediction.shape[1], prediction.shape[2], 1).to(prediction.device), prediction[:, :, :, :-1]), 3) 
        # shift_up = torch.cat((prediction[:, :, 1:, :], torch.zeros(1, prediction.shape[1], 1, prediction.shape[3]).to(prediction.device)), 2) 
        # shift_down = torch.cat((torch.zeros(1, prediction.shape[1], 1, prediction.shape[3]).to(prediction.device), prediction[:, :, :-1, :]), 2) 

        # return (prediction > 0.01) * (prediction > shift_left) * (prediction > shift_right) * (prediction > shift_up) * (prediction > shift_down)
        return prediction > 0.02

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                            groups=in_channels
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=in_channels)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=1,
                            padding=0,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


    def weighted_binary_cross_entropy(self, output, target, weights=None):        
        if weights is not None:
            assert len(weights) == 2
            
            loss = weights[1] * (target * torch.log(output + 1e-8)) + \
                   weights[0] * ((1 - target) * torch.log(1 - output + 1e-8))
        else:
            loss = target * torch.log(output + 1e-8) + (1 - target) * torch.log(1 - output + 1e-8)

        return torch.neg(torch.mean(loss))


    def load_pretrain(self, state_dict):  # pylint: disable=redefined-outer-name
        """Loads the pretrained keys in state_dict into model"""
        for key in state_dict.keys():
            try:
                _ = self.load_state_dict({key: state_dict[key]}, strict=False)
                print("Loaded {} (shape = {}) from the pretrained model".format(
                    key, state_dict[key].shape))
            except Exception as e:
                print("Failed to load {}".format(key))
                print(e)
