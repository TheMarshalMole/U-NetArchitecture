#!/usr/bin/python3
"""
    This file contains general functions, like DataLoader, etc.
"""
import torch
import torch.utils.data as data
from PIL import Image
from glob import glob
import os, numpy
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from .configuration import bdd100k

class DataLoaderSegmentation(data.Dataset):
    def __init__(self, trainData, maskData, maskExtension, device, transform = None, shuffle = True, resize = None, numberClasses = 1):
        super(DataLoaderSegmentation, self).__init__()
        self.__device = device
        self.__transform = transform
        self.__resize = resize
        self.__maskExtension = maskExtension
        self.numberClasses = numberClasses

        # read the paths
        self.__imgFiles = glob(os.path.join(trainData, '*.jpg'))
        if shuffle == 3:
            numpy.random.shuffle(self.__imgFiles)

        self.__maskFiles = []
        for imgPath in self.__imgFiles:
            newPath = '{}.{}'.format(imgPath.split('.')[1],  self.__maskExtension)
            mPath = os.path.join(maskData, os.path.basename(newPath))
            self.__maskFiles.append(mPath)
    
    def __BreakColorLabelIntoClasses(self, imgs, dictionary: dict):
        """
            It will analyze a color label and break it into N channels, each channel for a category.

            @params imgs the image which will be processed
            @params dictionary the dictionary containing the association between color - id

            @returns numpyarray with N channel of size of the initial image
        """
        # this will keep the track of color / category for the given color
        imgs = numpy.asarray(imgs)
        processedData = numpy.empty(shape=(512, 512))
        for x in range(512):
            for y in range(512):
                color = tuple(imgs[x][y].tolist())
                index = dictionary[color]
                processedData[x][y] = index
        return processedData

    def __Preprocess(self, imgs, masks, indexSelect):
        # BatchSize, Height, Width, Channels 
        data = numpy.empty(shape = (1, 512, 512, 3))
        if self.numberClasses == 1:
            label = numpy.empty(shape = (1, 512, 512, 1))
        else:
            label = numpy.empty(shape = (1, 512, 512))
        
        # obtain the dictionary to process data
        if self.numberClasses > 1:
            colorCategory = bdd100k.GetColorCategory()

        # only one selection, so transform it to vector
        if type(indexSelect) == int:
            imgs = [imgs[indexSelect], ]
            masks = [masks[indexSelect], ]
        else:
            imgs = imgs[indexSelect]
            masks = masks[indexSelect]

        for index in range(len(imgs)):
            # read data from disk
            _data = Image.open(imgs[index])
            _label = Image.open(masks[index])

            # resize data
            if self.__resize is not None:
                _data = _data.resize(self.__resize)
                _label = _label.resize(self.__resize)

            # transform to numpy and add a new axis
            _data = numpy.asarray(_data)[numpy.newaxis, ...]

            if self.numberClasses == 1:
                _label = numpy.expand_dims(_label, axis=2)
            else:
                _label = self.__BreakColorLabelIntoClasses(imgs = _label, dictionary = colorCategory)
            _label = numpy.asarray(_label)[numpy.newaxis, ...]

            # normalize data
            if _data.max() > 1:
                _data = _data / 255

            data = numpy.concatenate((data, _data), axis=0)
            label = numpy.concatenate((label, _label), axis=0)

        # PyTorch data format (BatchSize, Channels, Height, Width)
        dTorch = torch.from_numpy(data[1:]).float().permute(0, 3, 1, 2)
        if self.numberClasses == 1:
            lTorch = torch.from_numpy(label[1:]).float().permute(0, 3, 1, 2)
        else:
            lTorch = torch.from_numpy(label[1:]).float()
            
        # transform data
        if self.__transform is not None:
            dTorch = self.__transform(dTorch)
            lTorch = self.__transform(lTorch)
    
        dTorch = dTorch.to(device=self.__device)
        lTorch = lTorch.to(device=self.__device)
        # save data
        return dTorch, lTorch

    def GetName(self, index):
        return self.__imgFiles[index]

    def SaveImage(self, image):
        #image = image.transpose((3, 2, 1, 0))
        image = image[:, :, :]
        plt.imsave('result.png', image[:, :, 0])

    def __getitem__(self, index):
        """
            This function is triggered when an array it is accesed. It will return the correct
            format for the batch processing.

            @params index is a slice object or an int. The function should be careful to deal
            with both situations

            @retuns a good tuple consisting in 2 PyTorch objects,
        """
        # postprocess the data
        return self.__Preprocess(self.__imgFiles, self.__maskFiles, index)

    def __len__(self):
        return len(self.__imgFiles)

def Transformation(tensor):
    """
        Currently it supports only flipping
    """
    rValue = random.randint(0, 5)
    if rValue == 5:
        tensor = tensor.flip(1)
    elif rValue == 3:
        tensor = tensor.flip(3)
    elif rValue == 2:
        tensor = tensor.flip(3).flip(1)
    return tensor