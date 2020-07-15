# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:47:15 2020
@author: danie
"""

import numpy as np
import nibabel as nib
import csv
import itertools
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import glob
import pathlib
from pathlib import Path
from imresize import *

# Trains on a set of upsampled images to predict fiducial location at a high level.
current = pathlib.Path('fine_train.py').parent.absolute()
os.chdir(current)
os.chdir('/project/6050199/dcao6/autofid/')
os.chdir('pythonimg')

nii_list = []
fcsv_list = []
for file in glob.glob('sub-0***_T1w_rigid.nii.gz'):
    nii_list.append(file)

os.chdir('..')
os.chdir('pythonfcsv')

for file in glob.glob('OAS1-0***_MR1_T1_MEAN_mni_rigid.fcsv'):
    fcsv_list.append(file)
    
print(nii_list)
print(fcsv_list)
len(nii_list)
os.chdir('..')


# Loops through for each of 32 fiducials.
for g in range(32):
    finalpredarr = np.zeros((1,2001))
    for i in range(len(nii_list)):
        # Loading image.
        os.chdir('/project/6050199/dcao6/autofid/')
        os.chdir('pythonimg')          
        niimeta = nib.load(nii_list[i])
        hdr = niimeta.header
        img = niimeta.get_fdata()
        img = np.transpose(img,(2,0,1))
        
        # Loading and processing .fcsv file.
        os.chdir('..')
        os.chdir('pythonfcsv')         
        with open(fcsv_list[i]) as file:
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
    

                ijk = arr[g].reshape(-1,1)
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
                fill = np.matmul(four,trying[g].reshape(-1,1))
                fill = fill.reshape(4)
                newarr.append(fill)
                
                arr = np.array(newarr)
                
                arr = arr-1
                
                print(arr)
            
            else:
                print('Error in sform_code or qform_code, cannot obtain coordinates.')                    
                

        os.chdir('..')
        img = np.single(img)
        img = (img-np.amin(img))/(np.amax(img)-np.amin(img))
        
        arr = np.rint(arr)
        arr = arr.astype(int)
        perm = [2,0,1]
        arr = arr[:,perm]
        
        patch = img[arr[0,0]-30:arr[0,0]+31,arr[0,1]-30:arr[0,1]+31,arr[0,2]-30:arr[0,2]+31]
        
        # Upsampled image patch (only using patch because other parts of image are not relevant).
        patch = imresize(patch,2)

        if arr[0,0] < 30 or arr[0,1] < 30 or arr[0,2] < 30:
            print('skip')
            continue

        J = patch.cumsum(0).cumsum(1).cumsum(2)

        iterables = [ range(60-5,60+6), range(60-5,60+6), range(60-5,60+6) ]

        inner = []
        outer = []

        for t in itertools.product(*iterables):
            inner.append(t)

        iterables = [ range(60-10,60+11,2), range(60-10,60+11,2), range(60-10,60+11,2) ]

        for t in itertools.product(*iterables):
            outer.append(t)

        inner = np.array(inner)
        outer = np.array(outer)

        full = np.concatenate((inner,outer))
        full = np.unique(full,axis=0)
 
        # Loads offset file that specifies where to extract features.
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
        cornerlist = cornerlist.astype(int)

        Jnew = np.zeros((J.shape[0]+1,J.shape[1]+1,J.shape[2]+1))
        Jnew[1:,1:,1:] = J
        J = Jnew

        testerarr = np.zeros((4000*full.shape[0]))
        
        # Generation of features (random blocks of intensity around fiducial)
        numerator = J[cornerlist[:,3] + 1, cornerlist[:,4]+1, cornerlist[:,5]+1] - J[cornerlist[:,3]+1,cornerlist[:,4]+1,cornerlist[:,2]] - \
        J[cornerlist[:,3]+1,cornerlist[:,1],cornerlist[:,5]+1] - J[cornerlist[:,0],cornerlist[:,4]+1,cornerlist[:,5]+1] + \
        J[cornerlist[:,0],cornerlist[:,1],cornerlist[:,5]+1] + J[cornerlist[:,0],cornerlist[:,4]+1,cornerlist[:,2]] + \
        J[cornerlist[:,3]+1,cornerlist[:,1],cornerlist[:,2]] - J[cornerlist[:,0],cornerlist[:,1],cornerlist[:,2]]

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
        diff = np.reshape(diff,(full.shape[0],2000))
        dist = full - 60
        p = np.sqrt(dist[:,0]**2 + dist[:,1]**2 + dist[:,2]**2)

        finalpred = []
        for index in range(p.shape[0]):
            finalpred.append(np.hstack((diff[index],p[index])))

        finalpred = np.asarray(finalpred)
        # Concatenate to array of feature vectors.
        finalpredarr = np.concatenate((finalpredarr,finalpred))

    # Model training.
    finalpredarr = finalpredarr[1:,:]  
    print('training start')    
    regr_rf = RandomForestRegressor(n_estimators=20, max_features=0.33, min_samples_leaf=5,
                                random_state=2, n_jobs=-1)
    X_train = finalpredarr[:,:-1]
    y_train = finalpredarr[:,-1]
    Mdl = regr_rf.fit(X_train, y_train)
    
    model2save = 'finemodelfid{}'.format(g+1)
    os.chdir(current)
    Path("models").mkdir(parents=True, exist_ok=True)
    os.chdir('models')
    with open(model2save, 'wb') as f:
        joblib.dump(Mdl, f)
    
    os.chdir('..')
    print('complete')

