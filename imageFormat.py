#write the script for taking downloaded images and formatting them into pickle for the load_data function.

'''
IMPORTANT NOTE: The data file is saved to the wrong location for use by the NN, needs to be
moved to DogProjScripts.
'''

import numpy as np
import matplotlib as plt
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False
import glob
import os
import random
import pylab
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pickle
from scipy import ndimage
from scipy.misc import imresize
from skimage.transform import rescale, resize 

def cropImages(allFilePaths=None,imgCounter2=0,imagesRejected=0,finalSize = 200,
    trainingNumber = 2250,validationNumber = 140,testNumber = 260):
    #print(allFilePaths)
    training_x = np.zeros([trainingNumber,finalSize*finalSize])
    training_y = np.zeros([trainingNumber])
    validation_x =np.zeros([validationNumber,finalSize*finalSize])
    validation_y = np.zeros([validationNumber])
    test_x = np.zeros([testNumber,finalSize*finalSize])
    test_y = np.zeros([testNumber])
    if len(allFilePaths) > 1:
        print('ok')
    else:
        allFilePaths = (allFilePaths,allFilePaths)
    for filepath in allFilePaths:
        print('lll')
        #print(filepath)
        #if numberSuccessStacked >= trainSetSize or numberStacked == imageSetSize: 
         #   break
        #numberStacked += 1
        #numberSuccessStacked += 1
        
        
        
        #fileNameStringWithExtension = os.path.basename(filepath)
        #fileNameString = os.path.splitext(fileNameStringWithExtension)[0]
        #maskPath = os.path.join(path, 'masks/'+fileNameString+'_mask')
        
        
        
        '''
        try:
            maskRaw = Image.open(maskPath+'.jpg')
        except IOError:
            print('Image '+fileNameString+' has no corresponding mask, it has been skipped')
            numberSuccessStacked -= 1
            continue
        '''
        print('bla')
        #print(filepath)
        try:
            im = Image.open(filepath).convert(mode='L')#.convert('1')
        except:
            print('image ' + str(imgCounter2) + ' is truncated and not loaded')
            imagesRejected += 1
        im = np.asarray(im)
        im = (im/255).astype(float)
        #plt.imshow(im, cmap = cm.Greys_r)
        #plt.show()
        im = im[0:(im.shape[0]-(im.shape[0]*0.1)),...]#cropping 10% off bottom of image to remove most footers
        if im.shape[0]>=im.shape[1]:
            print('0pppp')
            shrinkRatio = im.shape[1]/finalSize
            im = resize(im,[int((1/shrinkRatio)*im.shape[0]),finalSize])
            margin = (im.shape[0]-finalSize)/2
            finalImage = im[margin:im.shape[0]-margin,...]
            print(finalImage.shape)
            print(margin)
            print(im.shape[0])
            #emptyImage = np.zeros([finalSize,finalSize])
            #emptyImage[:,margin:finalSize-margin] = emptyImage[:,margin:finalSize-margin] + im
            #finalImage = emptyImage
            
            
            
        else:
            print('1')
            shrinkRatio = im.shape[0]/finalSize
            a=int((1/shrinkRatio)*finalSize)
            #plt.imshow(im)
            #plt.show()
            im = resize(im,[finalSize,int((1/shrinkRatio)*im.shape[1])])
            #im = resize(im,1/shrinkRatio)
            margin = (im.shape[1]-finalSize)/2
            #emptyImage = np.zeros([finalSize,finalSize])
            #emptyImage[margin:finalSize-margin,:] = emptyImage[margin:finalSize-margin,:] + im
            #finalImage = emptyImage
            finalImage = im[:,margin:im.shape[1]-margin]
        normed = (finalImage - finalImage.mean()) / finalImage.std()   
        print(imgCounter2)
        #plt.imshow(finalImage, cmap = cm.Greys_r)
        #plt.show()
        #plt.imshow(normed, cmap = cm.Greys_r)
        #plt.show()
        print('flattening section')
        if imgCounter2 < int(trainingNumber/2):
            print('training pic true')
            training_x[imgCounter2][:] = finalImage.flatten()
            training_y[imgCounter2] = 1
            
        elif imgCounter2 < int(trainingNumber):
            print('training pic false')
            
            training_x[imgCounter2][:] = finalImage.flatten()
            training_y[imgCounter2] = 0
        elif imgCounter2 < int(trainingNumber+validationNumber/2):
            print('validation pic true')
            validation_x[imgCounter2-trainingNumber][:] = finalImage.flatten()
            validation_y[imgCounter2-trainingNumber] = 1
        elif imgCounter2 < int(trainingNumber+validationNumber):
            print('validation pic false')
            validation_x[imgCounter2-trainingNumber][:] = finalImage.flatten()
            validation_y[imgCounter2-trainingNumber] = 0
        elif imgCounter2 < int(trainingNumber+validationNumber+testNumber/2):
            print('test pic true')
            test_x[imgCounter2-(trainingNumber+validationNumber)][:] = finalImage.flatten()
            test_y[imgCounter2-(trainingNumber+validationNumber)] = 1
        elif imgCounter2 < int(trainingNumber+validationNumber+testNumber):
            print('test pic false')
            test_x[imgCounter2-(trainingNumber+validationNumber)][:] = finalImage.flatten()
            test_y[imgCounter2-(trainingNumber+validationNumber)] = 0
        else:
            print('concatenated training, validation and test sets then \
            exited, although there are more photos')
            break
        imgCounter2 += 1
    print('return')
    return training_x,training_y,validation_x,validation_y,test_x,test_y
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
if __name__ == '__main__':
    #path = './palstaves2/2013T482_Lower_Hardres_Canterbury/Axe1/'
    #levels = 5
    globalPath = r'C:\Users\Matt\Desktop\DogProj\ButterflyPhotos'
    #outputFilename = os.path.join(path,'trainingData'+str(SAMPLE_NUMBER)+'.npz')
    finalSize = 200
    imageSetSize = 0
    trainRatio = 4
    trainingNumber = 2250#1800#1800 # actually 237 / 143 beetles/// now 1200 True, 1100 False// now 1400 true, 1300 false
    validationNumber = 140#30#260#60 #/30 of each / now 260 tot
    testNumber = 260#300#100 #/70 of each / now 300 tot

    training_x = np.zeros([trainingNumber,finalSize*finalSize])
    training_y = np.zeros([trainingNumber])
    validation_x =np.zeros([validationNumber,finalSize*finalSize])
    validation_y = np.zeros([validationNumber])
    test_x = np.zeros([testNumber,finalSize*finalSize])
    test_y = np.zeros([testNumber])

    jpgFilePaths = []
    pngFilePaths = []
    allFilePaths = []
    imgCounter1 = 0
    while imgCounter1 < trainingNumber+validationNumber+testNumber:
        if imgCounter1 < int(trainingNumber/2):
            path = os.path.join(globalPath, 'training_BTrue')
            jpgFilePathsNew = (glob.glob(os.path.join(path, '*.jpg')))
            pngFilePathsNew = (glob.glob(os.path.join(path, '*.png')))
            bothFilePathsNew = (jpgFilePathsNew + pngFilePathsNew)
            assert (len(bothFilePathsNew) >= trainingNumber/2),"Not enough True training pics"
            bothFilePathsNew = bothFilePathsNew[0:int(trainingNumber/2)]
            allFilePaths = allFilePaths + bothFilePathsNew
            print(str(len(allFilePaths))+'trainTrue')
        elif imgCounter1 < trainingNumber:
            path = os.path.join(globalPath, 'training_BFalse')
            jpgFilePathsNew = (glob.glob(os.path.join(path, '*.jpg')))
            pngFilePathsNew = (glob.glob(os.path.join(path, '*.png')))
            bothFilePathsNew = (jpgFilePathsNew + pngFilePathsNew)
            assert (len(bothFilePathsNew) >= trainingNumber/2),"Not enough False training pics"
            bothFilePathsNew = bothFilePathsNew[0:int(trainingNumber/2)]
            allFilePaths = allFilePaths + bothFilePathsNew
            print(str(len(allFilePaths))+'trainFalse')
        elif imgCounter1 < int(trainingNumber+(validationNumber)/2):
            path = os.path.join(globalPath, 'validation_BTrue')
            jpgFilePathsNew = (glob.glob(os.path.join(path, '*.jpg')))
            pngFilePathsNew = (glob.glob(os.path.join(path, '*.png')))
            bothFilePathsNew = (jpgFilePathsNew + pngFilePathsNew)
            print(imgCounter1)
            print(len(bothFilePathsNew))
            assert (len(bothFilePathsNew) >= int((validationNumber)/2)),"Not enough True validation pics"
            bothFilePathsNew = bothFilePathsNew[0:int((validationNumber)/2)]
            allFilePaths = allFilePaths + bothFilePathsNew
            print(str(len(allFilePaths))+'validTrue')
        elif imgCounter1 < trainingNumber+validationNumber:
            path = os.path.join(globalPath, 'validation_BFalse')
            jpgFilePathsNew = (glob.glob(os.path.join(path, '*.jpg')))
            pngFilePathsNew = (glob.glob(os.path.join(path, '*.png')))
            bothFilePathsNew = (jpgFilePathsNew + pngFilePathsNew)
            print(len(bothFilePathsNew))
            assert (len(bothFilePathsNew) >= int((validationNumber/2))),"Not enough False validation pics"
            bothFilePathsNew = bothFilePathsNew[0:int(validationNumber/2)]
            allFilePaths = allFilePaths + bothFilePathsNew
            print(str(len(allFilePaths))+'validFalse')
        elif imgCounter1 < int((trainingNumber+validationNumber)+(testNumber)/2):
            path = os.path.join(globalPath, 'testing_BTrue')
            jpgFilePathsNew = (glob.glob(os.path.join(path, '*.jpg')))
            pngFilePathsNew = (glob.glob(os.path.join(path, '*.png')))
            bothFilePathsNew = (jpgFilePathsNew + pngFilePathsNew)
            assert (len(bothFilePathsNew) >= int(((testNumber)/2))),"Not enough True test pics"
            bothFilePathsNew = bothFilePathsNew[0:int((testNumber)/2)]
            allFilePaths = allFilePaths + bothFilePathsNew
            print(str(len(allFilePaths))+'testTrue')
        elif imgCounter1 < (trainingNumber+validationNumber+testNumber):
            path = os.path.join(globalPath, 'testing_BFalse')
            jpgFilePathsNew = (glob.glob(os.path.join(path, '*.jpg')))
            pngFilePathsNew = (glob.glob(os.path.join(path, '*.png')))
            bothFilePathsNew = (jpgFilePathsNew + pngFilePathsNew)
            assert (len(bothFilePathsNew) >= int((testNumber/2))),"Not enough False test pics"
            bothFilePathsNew = bothFilePathsNew[0:int(testNumber/2)]
            allFilePaths = allFilePaths + bothFilePathsNew
            print(str(len(allFilePaths))+'testFalse')
           
        else:
            print('error in imgCounter1 loop')
        imgCounter1 = len(allFilePaths)
        print('here')
        print(len(jpgFilePaths + pngFilePaths))    
    print(imgCounter1)
    #jpgFilePaths = glob.glob(os.path.join(path, '*.jpg'))
    #pngFilePaths = glob.glob(os.path.join(path, '*.png'))

    imagesRejected = 0
    imgCounter2 = 0
    #random.shuffle(shuffled)
    print(len(allFilePaths))
    print('oji')

    training_x,training_y,validation_x,validation_y,test_x,test_y=cropImages(allFilePaths) 
    #cropImages()   


    training_x,training_y = shuffle_in_unison(training_x,training_y)
    #validation_x,validation_y = shuffle_in_unison(validation_x,validation_y)
    #test_x,test_y = shuffle_in_unison(test_x,test_y)

    print('imagesRejected = ' + str(imagesRejected))
    outputFilename = os.path.join(globalPath,'dataset1'+'.npz')
    #np.savez_compressed(outputFilename,training_x=training_x,training_y=training_y, \
    #validation_x=validation_x,validation_y=validation_y,test_x=test_x,test_y=test_y,)
    dataset = [(training_x,training_y),(validation_x,validation_y),(test_x,test_y)]
    globalPath = r'C:\Users\Matt\Desktop\DogProj\scripts\DogProjScripts'
    basePath = r'C:\Users\Matt\Desktop\DogProj\data'
    print(type(test_x))
    pickle.dump( dataset, open( os.path.join(basePath,"dataset3.pkl"), "wb" ) ) # needed


    '''
    #imArray = im
    print('here2')
    maskArray = np.asarray(maskRaw) #not all 255 or 0 because of compression, may need to threshold

    maskArray = resize(maskArray,[totalSob.shape[0],totalSob.shape[1]])
    maskArray *= 255
    flatMaskArray = maskArray.reshape(maskArray.shape[0]*maskArray.shape[1])
    flatImArray = imArray.reshape(imArray.shape[0]*imArray.shape[1],imArray.shape[2])

    foreGround = (flatMaskArray>=64)
    backGround = (flatMaskArray<64)
    foreGroundSamples = flatImArray[foreGround,...]
    backGroundSamples = flatImArray[backGround,...]

    maxSampleCount = min(foreGroundSamples.shape[0],backGroundSamples.shape[0])
    outputSampleCount = min(maxSampleCount,int(SAMPLE_NUMBER))
    foreGroundIndices = np.random.choice(foreGroundSamples.shape[0],replace=False,size=outputSampleCount)
    backGroundIndices = np.random.choice(backGroundSamples.shape[0],replace=False,size=outputSampleCount)

    X = np.vstack([foreGroundSamples[foreGroundIndices,...],backGroundSamples[backGroundIndices,...]])
    y = np.concatenate([np.ones(outputSampleCount),np.zeros(outputSampleCount)])

    #wholeArray = np.array(joinedArray.shape[0],joinedArray.shape[1])
    #print(wholeArray.shape)
    #a=joinedArray[0:1000000].reshape(4,1000000)
    #print(a.shape)
    wholeXArray = np.concatenate((wholeXArray,X),axis=0)
    wholeyArray = np.concatenate((wholeyArray,y),axis=0)
    np.savez_compressed(outputFilename,X=wholeXArray,y=wholeyArray,S=int(SAMPLE_NUMBER),R=trainRatio,shuffled=shuffled)
    print('Stacked image '+fileNameString+ '; number '+str(numberSuccessStacked)+' out of '+str(trainSetSize))
    '''