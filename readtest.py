# coding:UTF-8
'''
Created on 2015年5月12日
@author: zhaozhiyong
'''

import scipy.io as scio

dataFile = 'data/1/corrected_dose.mat'
data = scio.loadmat(dataFile)
print(data.shape)