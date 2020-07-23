#!/usr/bin/env python
# coding: utf-8

# In[62]:

import csv
import numpy as np
import sys

arg1 = snakemake.input[0]
arg2 = snakemake.input[1]
arg3 = snakemake.input[3]
arg4 = snakemake.output[0]

# In[113]:


with open(arg1, 'r') as file:
    reader = csv.reader(file)
    next(reader)
    next(reader)
    next(reader)
    arr = np.empty((0,3))
    for row in reader:
        x = row[1:4]
        arr = np.vstack([arr,x])
    arr = np.asarray(arr,dtype='float64')


# In[115]:


f = open(arg2, 'r')
contents = f.readlines()
list_of_lists = []
for line in contents:
    stripped_line = line.strip()
    line_list = stripped_line.split()
    list_of_lists.append(line_list)

tform = np.empty((0,4))
for row in list_of_lists:
    x = row[:]
    tform = np.vstack([tform,x])

tform = np.asarray(tform,dtype='float64')


# In[114]:

tform = np.linalg.inv(tform)
ones = np.ones((32,1))
arr = np.hstack((arr,ones))

'''
factor = np.array([[1, 1, 1, 1],[1, 1, 1, 1],[1, 1, 1, 1],[1, 1, 1, 1]])
tform = np.multiply(tform,factor)
'''

# In[145]:


tform_applied = np.empty((0,4))
for i in range(32):
    x = np.matmul(tform,arr[i].transpose())
    tform_applied = np.vstack([tform_applied,x])



# In[164]:


with open(arg3, 'r') as file:
    list_of_lists = []
    reader = csv.reader(file)
    for i in range(3):
        list_of_lists.append(next(reader))
    for idx, val in enumerate(reader):
        val[1:4] = tform_applied[idx][:3]
        list_of_lists.append(val)


# In[167]:


with open(arg4, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(list_of_lists)





