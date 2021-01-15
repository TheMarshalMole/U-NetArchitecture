#!/usr/bin/python3
"""
    This file contains components for our U-Net network. You can see the architecture presented
    here: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/.
    The implementation is written in pytorch.

    Informations:
    The architecture consists of two major components:
    - encoder - which extracts feature from the image
    - decoder - which tries to decode the processed encodings
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import OrderedDict 

class DoubleConvolution(nn.Module):
    """
        This component describes the double convolution used for the contracting part.
    """
    def __init__(self, inpChannel: int, outChannel: int):
        """
            @params inpChannel - the number of channels for input
            @params outChannel - the number of channels for output
        """
        super(DoubleConvolution, self).__init__()
        self.__convLayer = nn.Sequential(
            OrderedDict([
                # first double convolution layer
                ('convLayer1', nn.Conv2d(inpChannel, outChannel, kernel_size=3, padding=1)),
                ('batchNormal1', nn.BatchNorm2d(outChannel)),
                ('relu1', nn.ReLU(inplace=True)),
                ('convLayer2', nn.Conv2d(outChannel, outChannel, kernel_size=3, padding=1)),
                ('batchNormal2', nn.BatchNorm2d(outChannel)),
                ('relu2', nn.ReLU(inplace=True)),
            ])
        )
    
    def forward(self, imgSrc):
        """
            Start forward propagation for the image

            @params imgSrc - the image read in a numpy format, or a feature map

            @returns a feature map after all of the propagation
        """
        return self.__convLayer(imgSrc)


def ConcatenateImages(imgSrc, imgAux):
    """
        This function will concatenate the imgSrc to imgAux keeping the size of the
        second image

        @params imgSrc first image to concatenate, as a tensor
        @params imgAux second image to concatenate, as a tensor

        @returns the concatenation result
    """
    diffY = imgSrc.size()[2] - imgAux.size()[2]
    diffX = imgSrc.size()[3] - imgAux.size()[3]

    imgSrc = f.pad(imgSrc, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
    return torch.cat([imgAux, imgSrc], dim=1)

class DiceCoeffiencyOriginal(nn.Module):
    """
        This method will compare 2 images provided as input and compute the dice coefficient on them
        https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

        @params imgSrc - first image
        @params imgAux - second image

        @returns dice coefficient as an int
    """
    def __init__(self):
        super(DiceCoeffiencyOriginal, self).__init__()

    def forward(self, imgSrc, imgAux):
        """
            Original implementation
            intersection = torch.sum(imgSrc * imgAux) # se inmulteste element-wise
            A = torch.sum(imgSrc)
            B = torch.sum(imgAux)
            return 1 - (2 * intersection)/(A + B)
        """
        smooth = 1.

        # have to use contiguous since they may from a torch.view op
        iflat = imgSrc.contiguous().view(-1)
        tflat = imgAux.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(tflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
