import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color

import handlePhs

header=handlePhs.readHeader('./header')

a=handlePhs.readPhs("./result7/S_6.sig")

b=np.round(a.reshape([header.length,header.width])).astype('int64')

def to255(array,length,width):

    ymax = 255
    ymin = 0
    xmax = max(map(max,array))
    xmin = min(map(min,array))
    
    for i in range(length):
        for j in range(width):
            array[i][j] = round(((ymax-ymin)*(array[i][j]-xmin)/(xmax-xmin))+ymin)
    
    return array

gray_image=to255(b,header.length,header.width)

d = [5]  # 距离
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 角度

# 计算GLCM
glcm = greycomatrix(gray_image, distances=d, angles=angles, levels=256, symmetric=True, normed=True)

# 计算GLCM特征
contrast = greycoprops(glcm, 'contrast')
dissimilarity = greycoprops(glcm, 'dissimilarity')
energy = greycoprops(glcm, 'energy')
correlation = greycoprops(glcm, 'correlation')

# 输出特征值
print("Contrast:", contrast)
print("Dissimilarity:", dissimilarity)
print("Energy:", energy)
print("Correlation:", correlation)



