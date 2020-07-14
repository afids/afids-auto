# -*- coding: utf-8 -*-
"""
Created on Fri May 22 02:18:14 2020

@author: danie
"""

import numpy as np
import nibabel as nib
import csv
import itertools
import os
import time
import joblib
import pathlib
from imresize import *
from nilearn.image import resample_img
import pandas as pd


# Autofid main script: use on new mri T1w image to find predicted fiducial location.

# Demo: uncomment the following code, and then run: autofid_main(image,ground_truth,fid)
# By doing this, we will obtain the predicted coordinates and AFLE for the anterior commissure (fid 1).
# current = pathlib.Path('autofid_main.py').parent.absolute()
# os.chdir(current)
# image = str(current) + '\\OAS1\\sub-0109_T1w.nii'
# ground_truth = str(current) + '\\OAS1\\OAS1_0109_MR1_T1_MEAN.fcsv'
# fid = 1

def autofid_main(image, ground_truth=None, fid=1):
    
    # Loading the image in, as well as some initial variables.
    testingarr = np.empty((0,3))
    distarr = np.empty((0,1))
    
    niimeta = nib.load(image)
    niimeta = resample_img(niimeta,target_affine=np.eye(3))
    hdr = niimeta.header
    img = niimeta.get_fdata()
    img = np.transpose(img,(2,0,1))
    
    
    if ground_truth==None:
        print('No ground truth fiducial file specified. Continuing with execution...')
    else:
        # Obtain fiducial coordinates given a Slicer .fcsv file.
        with open(ground_truth) as file:
            csv_reader = csv.reader(file, delimiter=',')
            next(csv_reader)
            next(csv_reader)
            next(csv_reader)
            arr = np.empty((0,3))
            for row in csv_reader:
                x = row[1:4]
                x = np.asarray(x,dtype='float64')
                arr = np.vstack([arr,x])
            
            if (hdr['qform_code'] > 0 and hdr['sform_code'] == 0):
                newarr = []
                B = hdr['quatern_b']
                C = hdr['quatern_c']
                D = hdr['quatern_d']
                A = np.sqrt(1 - B**2 - C**2 - D**2)
                
                R = [A**2+B**2-C**2-D**2, 2*(B*C-A*D), 2*(B*D+A*C)], [2*(B*C+A*D), A**2+C**2-B**2-D**2,2*(C*D+A*B)], [2*(B*D-A*C), 2*(C*D+A*B), A**2+D**2-C**2-B**2]
                R = np.array(R)
    
                for i in range(32):
                    ijk = arr[i].reshape(-1,1)
                    ijk[2] = ijk[2]*hdr['pixdim'][0]
                    pixdim = hdr['pixdim'][1],hdr['pixdim'][2],hdr['pixdim'][3]
                    pixdim = np.array(pixdim).reshape(-1,1)
                    fill = np.matmul(R,ijk)*pixdim+np.vstack([hdr['qoffset_x'],hdr['qoffset_y'],hdr['qoffset_z']])
                    fill = fill.reshape(3)
                    newarr.append(fill)
                    
                arr = np.array(newarr)
                arr = arr-1
            
                print(arr)
            
            elif hdr['sform_code'] > 0:
                
                newarr = []
                four = np.vstack([hdr['srow_x'],hdr['srow_y'],hdr['srow_z'],[0,0,0,1]])
                four = np.linalg.inv(four)
                trying = np.hstack([arr,np.ones((32,1))])
                for i in range(32):
                    fill = np.matmul(four,trying[i].reshape(-1,1))
                    fill = fill.reshape(4)
                    newarr.append(fill)
                
                arr = np.array(newarr)
                    
                arr = arr-1
                print(arr)
            
            else:
                print('Error in sform_code or qform_code, cannot obtain coordinates.')
    
    # Starting at a coarse resolution level of the image (downsampled by a factor of 4).
    img = np.single(img)
    img = (img-np.amin(img))/(np.amax(img)-np.amin(img))
    img_new = imresize(img,0.25)
    
    img_pad = np.pad(img_new, 50, mode='constant')
    
    imagefile = os.path.basename(image)
    extra = '_initialfeatures'


    if os.path.exists(imagefile[:-4]+'%s' % extra + '.npy') == True:
        # In this section, if an 'initial features' file exists, then that file will be loaded.
        # Loading this will load intensity features for that image and will speed up processing speed.
        print('Loading initial features...')
        diff_coarse = np.load(imagefile[:-4]+'%s' % extra + '.npy')
        iterables = [ range(50,img_pad.shape[1]-50,2), range(50,img_pad.shape[2]-50,2), range(50,img_pad.shape[0]-50,2) ]
        
        full = []
        
        for t in itertools.product(*iterables):
            full.append(t)
            
        full = np.asarray(full)
        full = np.unique(full,axis=0)
        perm = [2,0,1]
        full = full[:,perm]
        Jstored = img_pad.cumsum(0).cumsum(1).cumsum(2)
        print('Starting autofid...')
    else:
        # If an 'initial features' file does not exist, then this section will be executed, and
        # an 'initial features' file will be generated by the end. This section scans through the
        # image and extracts intensity information.
        print('Creating initial features...')
        iterables = [ range(50,img_pad.shape[1]-50,2), range(50,img_pad.shape[2]-50,2), range(50,img_pad.shape[0]-50,2) ]
        full = []
        for t in itertools.product(*iterables):
            full.append(t) 
        full = np.asarray(full)
        full = np.unique(full,axis=0)
        Jstored = img_pad.cumsum(0).cumsum(1).cumsum(2)
        file = np.load('feature_offsets.npz')
        smin = file['arr_0']
        smax = file['arr_1']
        perm = [2,0,1]
        full = full[:,perm]
        smin = smin[:,perm]
        smax = smax[:,perm]      
        mincornerlist = np.zeros((4000*full.shape[0], 3)).astype('uint8')
        maxcornerlist = np.zeros((4000*full.shape[0], 3)).astype('uint8')
        for index in range(full.shape[0]):
            mincorner = full[index] + smin
            maxcorner = full[index] + smax
            mincornerlist[index*4000:(index+1)*4000] = mincorner
            maxcornerlist[index*4000:(index+1)*4000] = maxcorner
        cornerlist = np.hstack((mincornerlist,maxcornerlist))
        Jnew = np.zeros((img_pad.shape[0]+1,img_pad.shape[1]+1,img_pad.shape[2]+1))
        Jnew[1:,1:,1:] = Jstored
        Jcoarse = Jnew
        testerarr = np.zeros(cornerlist.shape[0])
        numerator = Jcoarse[cornerlist[:,3]+1, cornerlist[:,4]+1, cornerlist[:,5]+1] - Jcoarse[cornerlist[:,0],cornerlist[:,4]+1,cornerlist[:,5]+1] - \
        Jcoarse[cornerlist[:,3]+1,cornerlist[:,4]+1,cornerlist[:,2]] - Jcoarse[cornerlist[:,3]+1,cornerlist[:,1],cornerlist[:,5]+1] + \
        Jcoarse[cornerlist[:,3]+1,cornerlist[:,1],cornerlist[:,2]] + Jcoarse[cornerlist[:,0],cornerlist[:,1],cornerlist[:,5]+1] + \
        Jcoarse[cornerlist[:,0],cornerlist[:,4]+1,cornerlist[:,2]] - Jcoarse[cornerlist[:,0],cornerlist[:,1],cornerlist[:,2]]
        denominator = (cornerlist[:,3]-cornerlist[:,0]+1)*(cornerlist[:,4]-cornerlist[:,1]+1)*(cornerlist[:,5]-cornerlist[:,2]+1)
        testerarr = numerator/denominator
        
        vector1arr = np.zeros((4000*full.shape[0]))
        vector2arr = np.zeros((4000*full.shape[0]))
        for index in range(full.shape[0]):
            vector = range(index*4000,index*4000+2000)
            vector1arr[index*4000:(index+1)*4000-2000] = vector
        for index in range(full.shape[0]):
            vector = range(index*4000+2000,index*4000+4000)
            vector2arr[index*4000+2000:(index+1)*4000] = vector
        vector1arr[0] = 1
        vector1arr = vector1arr[vector1arr != 0]
        vector1arr[0] = 0
        vector2arr = vector2arr[vector2arr != 0]
        vector1arr = vector1arr.astype(int)
        vector2arr = vector2arr.astype(int)
        diff = testerarr[vector1arr] - testerarr[vector2arr]
        diff_coarse = np.reshape(diff,(full.shape[0],2000))
        np.save(imagefile[:-4]+'%s' % extra + '.npy',diff_coarse)
        print('Saving initial features file for future use...')
        print('Starting autofid...')
    
    # Loads offsets file. This file specifies the specific region around each fiducial 
    # where features are extracted. 
    file = np.load('feature_offsets.npz')
    smin = file['arr_0']
    smax = file['arr_1']
    perm = [2,0,1]
    smin = smin[:,perm]
    smax = smax[:,perm]
    os.chdir('..')
    os.chdir('training\\models')
    
    start = time.time()

    for g in range(fid-1,fid):
        
        # Loads regression forest trained on downsampled resolution, and tests the current image.
        with open('coarsemodelfid%d' % (g+1),'rb') as f:
            model = joblib.load(f)
        
        answer = model.predict(diff_coarse)
                
        df = pd.DataFrame(answer)
        
        idx = df[0].idxmin()
        testing = (full[idx]-50)*4
        
        iterables = [ range(full[idx][0]-3,full[idx][0]+4), range(full[idx][1]-3,full[idx][1]+4), range(full[idx][2]-3,full[idx][2]+4) ]
        
        full2 = []
        for t in itertools.product(*iterables):
            full2.append(t)
            
        full2 = np.asarray(full2)
        full2 = np.unique(full2,axis=0)
        
        mincornerlist = np.zeros((4000*full2.shape[0], 3)).astype('uint8')
        maxcornerlist = np.zeros((4000*full2.shape[0], 3)).astype('uint8')
        
        for index in range(full2.shape[0]):
            mincorner = full2[index] + smin
            maxcorner = full2[index] + smax
            mincornerlist[index*4000:(index+1)*4000] = mincorner
            maxcornerlist[index*4000:(index+1)*4000] = maxcorner
        
        cornerlist = np.hstack((mincornerlist,maxcornerlist))
        
        Jnew = np.zeros((Jstored.shape[0]+1,Jstored.shape[1]+1,Jstored.shape[2]+1))
        Jnew[1:,1:,1:] = Jstored
        Jcoarse = Jnew  
        
        testerarr = np.zeros((4000*full2.shape[0]))
        
        numerator = Jcoarse[cornerlist[:,3]+1, cornerlist[:,4]+1, cornerlist[:,5]+1] - Jcoarse[cornerlist[:,0],cornerlist[:,4]+1,cornerlist[:,5]+1] - \
        Jcoarse[cornerlist[:,3]+1,cornerlist[:,4]+1,cornerlist[:,2]] - Jcoarse[cornerlist[:,3]+1,cornerlist[:,1],cornerlist[:,5]+1] + \
        Jcoarse[cornerlist[:,3]+1,cornerlist[:,1],cornerlist[:,2]] + Jcoarse[cornerlist[:,0],cornerlist[:,1],cornerlist[:,5]+1] + \
        Jcoarse[cornerlist[:,0],cornerlist[:,4]+1,cornerlist[:,2]] - Jcoarse[cornerlist[:,0],cornerlist[:,1],cornerlist[:,2]]
        
        denominator = (cornerlist[:,3]-cornerlist[:,0]+1)*(cornerlist[:,4]-cornerlist[:,1]+1)*(cornerlist[:,5]-cornerlist[:,2]+1)
        
        testerarr = numerator/denominator
        
        vector1arr = np.zeros((4000*full2.shape[0]))
        vector2arr = np.zeros((4000*full2.shape[0]))
        
        for index in range(full2.shape[0]):
            vector = range(index*4000,index*4000+2000)
            vector1arr[index*4000:(index+1)*4000-2000] = vector
        
        for index in range(full2.shape[0]):
            vector = range(index*4000+2000,index*4000+4000)
            vector2arr[index*4000+2000:(index+1)*4000] = vector
        
        vector1arr[0] = 1
        vector1arr = vector1arr[vector1arr != 0]
        vector1arr[0] = 0
        vector2arr = vector2arr[vector2arr != 0]
        vector1arr = vector1arr.astype(int)
        vector2arr = vector2arr.astype(int)
        
        diff = testerarr[vector1arr] - testerarr[vector2arr]
        diff = np.reshape(diff,(full2.shape[0],2000))
        
        answer = model.predict(diff)
        df = pd.DataFrame(answer)
        
        idx = df[0].idxmin()
        
        testing = (full2[idx]-50)*4
        
        # Isolates area of high likelihood from previous model, then use a regression
        # forest trained on original resolution images to output a prediction.
        img_pad = np.pad(img, 50, mode='constant')
        
        testing = testing + 50

        patch = img_pad[testing[0]-60:testing[0]+61,testing[1]-60:testing[1]+61,testing[2]-60:testing[2]+61]
        patch = (patch-np.amin(patch))/(np.amax(patch)-np.amin(patch))
        Jstored2 = patch.cumsum(0).cumsum(1).cumsum(2)
        
        iterables = [ range(60-7,60+8,2), range(60-7,60+8,2), range(60-7,60+8,2) ]
        
        fullmed = []
        
        for t in itertools.product(*iterables):
            fullmed.append(t)
            
        fullmed = np.asarray(fullmed)
        fullmed = np.unique(fullmed,axis=0)
        
        perm = [2,0,1]
        fullmed = fullmed[:,perm]
        mincornerlist = np.zeros((4000*fullmed.shape[0], 3)).astype('uint8')
        maxcornerlist = np.zeros((4000*fullmed.shape[0], 3)).astype('uint8')
        
        for index in range(fullmed.shape[0]):
            mincorner = fullmed[index] + smin
            maxcorner = fullmed[index] + smax
            mincornerlist[index*4000:(index+1)*4000] = mincorner
            maxcornerlist[index*4000:(index+1)*4000] = maxcorner
        
        cornerlist = np.hstack((mincornerlist,maxcornerlist))
        
        Jnew = np.zeros((Jstored2.shape[0]+1,Jstored2.shape[1]+1,Jstored2.shape[2]+1))
        Jnew[1:,1:,1:] = Jstored2
        Jmed = Jnew  
        
        testerarr = np.zeros((4000*4913))
        
        numerator = Jmed[cornerlist[:,3]+1, cornerlist[:,4]+1, cornerlist[:,5]+1] - Jmed[cornerlist[:,0],cornerlist[:,4]+1,cornerlist[:,5]+1] - \
        Jmed[cornerlist[:,3]+1,cornerlist[:,4]+1,cornerlist[:,2]] - Jmed[cornerlist[:,3]+1,cornerlist[:,1],cornerlist[:,5]+1] + \
        Jmed[cornerlist[:,3]+1,cornerlist[:,1],cornerlist[:,2]] + Jmed[cornerlist[:,0],cornerlist[:,1],cornerlist[:,5]+1] + \
        Jmed[cornerlist[:,0],cornerlist[:,4]+1,cornerlist[:,2]] - Jmed[cornerlist[:,0],cornerlist[:,1],cornerlist[:,2]]
        
        denominator = (cornerlist[:,3]-cornerlist[:,0]+1)*(cornerlist[:,4]-cornerlist[:,1]+1)*(cornerlist[:,5]-cornerlist[:,2]+1)
        
        testerarr = numerator/denominator
        
        vector1arr = np.zeros((4000*fullmed.shape[0]))
        vector2arr = np.zeros((4000*fullmed.shape[0]))
        
        for index in range(fullmed.shape[0]):
            vector = range(index*4000,index*4000+2000)
            vector1arr[index*4000:(index+1)*4000-2000] = vector
        
        for index in range(fullmed.shape[0]):
            vector = range(index*4000+2000,index*4000+4000)
            vector2arr[index*4000+2000:(index+1)*4000] = vector
        
        vector1arr[0] = 1
        vector1arr = vector1arr[vector1arr != 0]
        vector1arr[0] = 0
        vector2arr = vector2arr[vector2arr != 0]
        vector1arr = vector1arr.astype(int)
        vector2arr = vector2arr.astype(int)
        
        diff = testerarr[vector1arr] - testerarr[vector2arr]
        diff = np.reshape(diff,(fullmed.shape[0],2000))
        
        with open('medmodelfid%d' % (g+1),'rb') as f:
            model = joblib.load(f)
        
        answer = model.predict(diff)
        df = pd.DataFrame(answer)
        
        idx = df[0].idxmin()
        
        iterables = [ range(fullmed[idx][0]-3,fullmed[idx][0]+4), range(fullmed[idx][1]-3,fullmed[idx][1]+4), range(fullmed[idx][2]-3,fullmed[idx][2]+4) ]
        
        fullmed = []
        
        for t in itertools.product(*iterables):
            fullmed.append(t)
            
        fullmed = np.asarray(fullmed)
        fullmed = np.unique(fullmed,axis=0)

        mincornerlist = np.zeros((4000*fullmed.shape[0], 3)).astype('uint8')
        maxcornerlist = np.zeros((4000*fullmed.shape[0], 3)).astype('uint8')
        
        for index in range(fullmed.shape[0]):
            mincorner = fullmed[index] + smin
            maxcorner = fullmed[index] + smax
            mincornerlist[index*4000:(index+1)*4000] = mincorner
            maxcornerlist[index*4000:(index+1)*4000] = maxcorner
        
        cornerlist = np.hstack((mincornerlist,maxcornerlist))
        
        Jnew = np.zeros((Jstored2.shape[0]+1,Jstored2.shape[1]+1,Jstored2.shape[2]+1))
        Jnew[1:,1:,1:] = Jstored2
        Jmed = Jnew  
        
        testerarr = np.zeros((4000*fullmed.shape[0]))
        
        numerator = Jmed[cornerlist[:,3]+1, cornerlist[:,4]+1, cornerlist[:,5]+1] - Jmed[cornerlist[:,0],cornerlist[:,4]+1,cornerlist[:,5]+1] - \
        Jmed[cornerlist[:,3]+1,cornerlist[:,4]+1,cornerlist[:,2]] - Jmed[cornerlist[:,3]+1,cornerlist[:,1],cornerlist[:,5]+1] + \
        Jmed[cornerlist[:,3]+1,cornerlist[:,1],cornerlist[:,2]] + Jmed[cornerlist[:,0],cornerlist[:,1],cornerlist[:,5]+1] + \
        Jmed[cornerlist[:,0],cornerlist[:,4]+1,cornerlist[:,2]] - Jmed[cornerlist[:,0],cornerlist[:,1],cornerlist[:,2]]
        
        denominator = (cornerlist[:,3]-cornerlist[:,0]+1)*(cornerlist[:,4]-cornerlist[:,1]+1)*(cornerlist[:,5]-cornerlist[:,2]+1)
        
        testerarr = numerator/denominator
        
        vector1arr = np.zeros((4000*fullmed.shape[0]))
        vector2arr = np.zeros((4000*fullmed.shape[0]))
        
        for index in range(fullmed.shape[0]):
            vector = range(index*4000,index*4000+2000)
            vector1arr[index*4000:(index+1)*4000-2000] = vector
        
        for index in range(fullmed.shape[0]):
            vector = range(index*4000+2000,index*4000+4000)
            vector2arr[index*4000+2000:(index+1)*4000] = vector
        
        vector1arr[0] = 1
        vector1arr = vector1arr[vector1arr != 0]
        vector1arr[0] = 0
        vector2arr = vector2arr[vector2arr != 0]
        vector1arr = vector1arr.astype(int)
        vector2arr = vector2arr.astype(int)
        
        diff = testerarr[vector1arr] - testerarr[vector2arr]
        diff = np.reshape(diff,(fullmed.shape[0],2000))
        
        answer = model.predict(diff)
        df = pd.DataFrame(answer)
        
        idx = df[0].idxmin()
        
    
        testing = fullmed[idx]-60+testing-50
        testing = np.rint(testing)
        testing = testing.astype('int')
        
        # Isolates area of high likelihood from previous model, then use a regression
        # forest trained on high resolution images to output final prediction.
        patch = img[testing[0]-30:testing[0]+31,testing[1]-30:testing[1]+31,testing[2]-30:testing[2]+31]
        patch = imresize(patch,2)
        
        Jstored3 = patch.cumsum(0).cumsum(1).cumsum(2)
    
        iterables = [ range(60-7,60+8), range(60-7,60+8), range(60-7,60+8) ]
    
        fullfine = []
    
        for t in itertools.product(*iterables):
            fullfine.append(t)
    
        fullfine = np.asarray(fullfine)
        fullfine = np.unique(fullfine,axis=0)
        

        perm = [2,0,1]
        fullfine = fullfine[:,perm]
        mincornerlist = np.zeros((4000*fullfine.shape[0], 3)).astype('uint8')
        maxcornerlist = np.zeros((4000*fullfine.shape[0], 3)).astype('uint8')
    
        for index in range(fullfine.shape[0]):
            mincorner = fullfine[index] + smin
            maxcorner = fullfine[index] + smax
            mincornerlist[index*4000:(index+1)*4000] = mincorner
            maxcornerlist[index*4000:(index+1)*4000] = maxcorner
    
        cornerlist = np.hstack((mincornerlist,maxcornerlist))
        cornerlist = cornerlist.astype(int)
    
        Jnew = np.zeros((Jstored3.shape[0]+1,Jstored3.shape[1]+1,Jstored3.shape[2]+1))
        Jnew[1:,1:,1:] = Jstored3
        Jfine = Jnew
    
        testerarr = np.zeros((4000*fullfine.shape[0]))
    
        numerator = Jfine[cornerlist[:,3] + 1, cornerlist[:,4]+1, cornerlist[:,5]+1] - Jfine[cornerlist[:,3]+1,cornerlist[:,4]+1,cornerlist[:,2]] - \
        Jfine[cornerlist[:,3]+1,cornerlist[:,1],cornerlist[:,5]+1] - Jfine[cornerlist[:,0],cornerlist[:,4]+1,cornerlist[:,5]+1] + \
        Jfine[cornerlist[:,0],cornerlist[:,1],cornerlist[:,5]+1] + Jfine[cornerlist[:,0],cornerlist[:,4]+1,cornerlist[:,2]] + \
        Jfine[cornerlist[:,3]+1,cornerlist[:,1],cornerlist[:,2]] - Jfine[cornerlist[:,0],cornerlist[:,1],cornerlist[:,2]]
    
        denominator = (cornerlist[:,3]-cornerlist[:,0]+1)*(cornerlist[:,4]-cornerlist[:,1]+1)*(cornerlist[:,5]-cornerlist[:,2]+1)
    
        testerarr = numerator/denominator
    
        vector1arr = np.zeros((4000*fullfine.shape[0]))
        vector2arr = np.zeros((4000*fullfine.shape[0]))
    
        for index in range(fullfine.shape[0]):
            vector = range(index*4000,index*4000+2000)
            vector1arr[index*4000:(index+1)*4000-2000] = vector
    
        for index in range(fullfine.shape[0]):
            vector = range(index*4000+2000,index*4000+4000)
            vector2arr[index*4000+2000:(index+1)*4000] = vector
    
        vector1arr[0] = 1
        vector1arr = vector1arr[vector1arr != 0]
        vector1arr[0] = 0
        vector2arr = vector2arr[vector2arr != 0]
        vector1arr = vector1arr.astype(int)
        vector2arr = vector2arr.astype(int)
    
        diff = testerarr[vector1arr] - testerarr[vector2arr]
        diff = np.reshape(diff,(fullfine.shape[0],2000))
        
        with open('finemodelfid%d' % (g+1),'rb') as f:
            model = joblib.load(f)
        
        answer = model.predict(diff)
        df = pd.DataFrame(answer)
        
        idx = df[0].idxmin()
        
        testing = testing - 30 + ((fullfine[idx]+0.5)/2)
        
        testingarr = np.vstack([testingarr,testing])
        if ground_truth != None:
            dist = np.sqrt((arr[g][0]-testing[1])**2 + (arr[g][1]-testing[2])**2 + (arr[g][2]-testing[0])**2)
            distarr = np.vstack([distarr,dist])
        print('Fid = ' + str(g+1))
    
    print('Fid coordinates = ' + str(testingarr[:]))
    if ground_truth != None:
        print('AFLE = ' + str(distarr[:]))
    end = time.time()
    elapsed = end - start
    print('Time to locate fiducial = ' + str(elapsed))
        
    os.chdir('..\\..')
    os.chdir('testing')
