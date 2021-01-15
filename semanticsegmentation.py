#!/usr/bin/python3
"""
    This file includes a Deep-Learning implementation of Semantic Segmentation.
    The architecture is: U-NET
"""
from library import UNetClassic, DataLoaderSegmentation, DiceCoeffiencyOriginal, Transformation
import torch
import matplotlib.pyplot as plt
import numpy as np
from os import walk, mkdir

import matplotlib.pyplot as plt

# for nice output
from tqdm import tqdm

def SaveImage(image, name='result'):
    imageTemp = image.detach().cpu()
    imageTemp = imageTemp[0, 0].permute(0, 1)
    plt.imsave('{}.png'.format(name), imageTemp)

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
            #transform = Transformation)
            transform = None,
            resize = None,
            numberClasses = 8
        )

def SaveNetwork(model, number):
    savePath = './saves/salvare{}'.format(number)
    torch.save(model.state_dict(), savePath)

def Normalize(image, threshold):
    """
        Data will be sent to a sigmoind function to map to extremes.

        @params image - the image which will be processed
        @params threshold - what are the boundaries

        @returns the processes image
    """
    image = torch.sigmoid(image)
    return image > threshold

def PlotData(data, name):
    plt.plot(data)
    plt.ylabel(name)
    plt.show()

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main():
    # select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    dataset = 'bdd100k'
    # size(batch), size(channel), height, width
    if dataset == 'kaggleCaravana':
        images = GetDataKaggleCaravana(device)
    elif dataset == 'bdd100k':
        images = GetDataBDD100k(device)
    numberOfClasses = images.numberClasses

    # model definition
    model = UNetClassic(numberClasses = numberOfClasses).to(device)
    model.apply(init_weights)
    model.train()

    # training data
    epochNumber = 1
    learnRate = 1e-3
    batchSize = 2

    # in case of -1 run all batches
    numberBatches = -1

    # frequency show info
    frequencyInfo = 25
    frequencySave = 500

    # testing
    testSize = 5

    # optimzere penru gradient
    # RMSprop: https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=learnRate, weight_decay=1e-8, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learnRate)
    if model.numberClasses > 1:
        # for multi-class please read: https://www.fer.unizg.hr/_download/repository/IMVIP_2017_HrkacBrkicKalafatic.pdf
        criterion = torch.nn.CrossEntropyLoss()
        maskType = torch.long
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
        maskType = torch.float32
    lossValues = []

    # dice coefficient
    diceCoeff = DiceCoeffiencyOriginal().to(device)
    diceResults = []
    def EvaluateNetwork(model, testSize):
        diceLost = 0
        for v in range(3500, 3500+testSize, 1):
            xTest, yTest = images[v]

            yExTest = model(xTest)
            for x in range(len(yExTest)):
                selResult = yExTest[x]
                selReal = yTest[x]

                # compute dice coeff
                selResult = Normalize(selResult, 0.3)
                diceLost += diceCoeff(selResult, selReal)

        diceResults.append(diceLost)

    # when to save a network
    # currently, the network is saved at every epoch
    
    """
        Used this github for reference: https://github.com/jcjohnson/pytorch-examples/blob/master/nn/two_layer_net_nn.py
    """
    for epoch in tqdm(range(epochNumber), desc='Epoch: '):
        # get data again, random
        if dataset == 'bdd100k':
            images = GetDataBDD100k(device)

        lostPerBatch = 0
        numberBatches = (len(images) // batchSize) - 1 if numberBatches == -1 else numberBatches
        for batch in tqdm(range(numberBatches), desc='Batch: '):
            # get all data from files
            xTrain, yExpected = images[batchSize * batch:batchSize * (batch+1)]

            # forward propagation
            yPredict = model(xTrain)
            yExpected = yExpected.to(maskType)
            # calculate loss
            lossValue = criterion(yPredict, yExpected)
            lostPerBatch += lossValue.item()

            optimizer.zero_grad()
            lossValue.backward()
            optimizer.step()

            # show statistics & save data
            if batch != 0 and batch%frequencyInfo == 0:
                if numberOfClasses == 1:
                    model.eval()
                    EvaluateNetwork(model, testSize)
                    model.train()

                # losses
                lossValues.append(lostPerBatch)
                lostPerBatch = 0
                print('Currently, the lost is:', lossValues)

                if batch % frequencySave == 0:
                    # intermediary savings
                    SaveNetwork(model, epoch * 100000 + batch)

        # save the network results
        SaveNetwork(model, epoch)

    # plot data part
    #PlotData(diceResults, "Dice Coefficient")
    PlotData(lossValues, "Loss Function")

    print(diceResults)
    print(lostPerBatch)

    xTest, _ = images[2001]
    yTest = model(xTest)
    SaveImage(yTest)

if __name__ == "__main__":
    main()