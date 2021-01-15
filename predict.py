#!/usr/bin/python3
"""
    This file includes a Deep-Learning implementation of Semantic Segmentation.
    The architecture is: U-NET
"""
from library import UNetClassic, DataLoaderSegmentation, DiceCoeffiencyOriginal
from library import bdd100k

import torch
import matplotlib.pyplot as plt

import numpy as np
from os import walk, mkdir

import matplotlib.pyplot as plt

# for nice output
from tqdm import tqdm

def GetDataKaggleCaravana(device):
    """
        This function will read the data from the file, as a numpy array.
        This is the data for the Kaggle Caravana

        @device for cuda
    """
    trainPath = './dataTraining/imgs'
    maskPath = './dataTraining/masks'

    return  DataLoaderSegmentation(
                trainData = trainPath, 
                maskData = maskPath,
                maskExtension = 'gif',
                device = device, 
                #transform = Transformation)
                shuffle = False,
                transform = None,
                resize = (512, 512),
                numberClasses = 1
            )

def GetDataBDD100k(device):
    """
        This function will read the data from the file, as a numpy array.
        This is the data for the BDD100k.

        @device for cuda
    """
    trainPath = './dataTraining/bdd100k/imgs'
    maskPath = './dataTraining/bdd100k/masks'

    return DataLoaderSegmentation(
            trainData = trainPath, 
            maskData = maskPath,
            maskExtension = 'png',
            device = device, 
            #transform = Transformation),
            shuffle = False,
            transform = None,
            resize = None,
            numberClasses = 8
        )

def SaveImage(image, name='result'):
    imageTemp = image.detach().cpu()
    imageTemp = imageTemp[0, 0].permute(0, 1)
    plt.imsave('{}.png'.format(name), imageTemp)

def SaveImageLayer(image, name='result'):
    imageTemp = image.detach().cpu()
    imageTemp = imageTemp.permute(0, 1)
    plt.imsave('{}.png'.format(name), imageTemp)

def SaveImageNumpy(image, name='result'):
    #imageTemp = image.transpose(2, 1, 0)
    plt.imsave('{}.png'.format(name), image)

def Normalize(image, threshold):
    """
        Data will be sent to a sigmoind function to map to extremes.

        @params image - the image which will be processed
        @params threshold - what are the boundaries

        @returns the processes image
    """
    image = torch.sigmoid(image)
    return image > threshold

def NormalizeSoftmax(image):
    """
        Data will be sent to a softmax for multiclass.

        @params image - the image which will be processed
        @params threshold - what are the boundaries

        @returns the processes image
    """
    maxim = torch.argmax(image[0], dim=0)
    return maxim

def main():
    # select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # size(batch), size(channel), height, width
    #images = GetDataKaggleCaravana(device)
    images = GetDataBDD100k(device)
    numberOfClasses = images.numberClasses

    # model to load
    #modelPath = './saves/salvare2450NAFinal'
    modelPath = './saves/salvare1000'
    #modelPath = './saves/salvare1050Aug'

    # foward propagation for results
    model = UNetClassic(numberClasses = numberOfClasses).to(device)

    # load data
    model.load_state_dict(torch.load(modelPath, map_location=device))

    #index = 2011
    #index = 4000
    #index = 3040
    index = 4000
    print('Apply on image: ', images.GetName(index))
    xTest, yExpected = images[index]
    yTest = model(xTest)
    
    if numberOfClasses > 1:
        SaveImageLayer(yTest[0][5], 'processed')
        yTest = NormalizeSoftmax(yTest)
        yTest = bdd100k.ReconstructImageByCategory(yTest)
        
        #yTest = bdd100k.ReconstructImageByCategory(yExpected)
        SaveImageNumpy(yTest)
    else:
        SaveImage(yTest, 'Original')
        yTest = Normalize(yTest, 0.5)
        SaveImage(yTest)
        SaveImage(yExpected, 'expected')

if __name__ == "__main__":
    main()