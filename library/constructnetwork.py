#!/usr/bin/python3
"""
    This module should contains all the methods necessary for constructing the network. The generated
    network is untrained.
"""
import torch
import torch.nn as nn
from . import DoubleConvolution, ConcatenateImages

class UNetClassic(nn.Module):
    """
        This class represent the classic U-Net Architecture described
        here: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    """
    def __init__(self, inputChannels=3, numberClasses=1):
        super(UNetClassic, self).__init__()

        # adding data to global
        self.numberClasses = numberClasses

        # contracting part
        self.__imgProcess0 = DoubleConvolution(inputChannels, 64)

        self.__imgProcess1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, ceil_mode=True),
            DoubleConvolution(64, 128)
        )
        self.__imgProcess2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, ceil_mode=True),
            DoubleConvolution(128, 256)
        )
        self.__imgProcess3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, ceil_mode=True),
            DoubleConvolution(256, 512)
        )
        self.__imgProcess4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, ceil_mode=True),
            DoubleConvolution(512, 1024)
        )

        # expanding part
        self.__upTrans0 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(2,2), stride=2)
        self.__upTrans1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2,2), stride=2)
        self.__upTrans2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2,2), stride=2)
        self.__upTrans3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2,2), stride=2)

        self.__imgProcessE3 = DoubleConvolution(1024, 512)
        self.__imgProcessE2 = DoubleConvolution(512, 256)
        self.__imgProcessE1 = DoubleConvolution(256, 128)
        self.__imgProcessE0 = DoubleConvolution(128, 64)

        # final convolution
        self.__finalConv = nn.Conv2d(64, numberClasses, kernel_size=1)
    

    def forward(self, imgSrc):
        """
            This will apply the algorithm on the input image
        """

        # contracting part
        img11 = self.__imgProcess0(imgSrc)
        img21 = self.__imgProcess1(img11)
        img31 = self.__imgProcess2(img21)
        img41 = self.__imgProcess3(img31)
        
        # bottom
        img51 = self.__imgProcess4(img41)

        # expanding part
        imgE41 = self.__upTrans0(img51)
        imgE42 = ConcatenateImages(img41, imgE41)
        imgE43 = self.__imgProcessE3(imgE42)

        imgE31 = self.__upTrans1(imgE43)
        imgE32 = ConcatenateImages(img31, imgE31)
        imgE33 = self.__imgProcessE2(imgE32)

        imgE21 = self.__upTrans2(imgE33)
        imgE22 = ConcatenateImages(img21, imgE21)
        imgE23 = self.__imgProcessE1(imgE22)

        imgE11 = self.__upTrans3(imgE23)
        imgE12 = ConcatenateImages(img11, imgE11)
        imgE13 = self.__imgProcessE0(imgE12)

        imgFinal = self.__finalConv(imgE13)
        return imgFinal
