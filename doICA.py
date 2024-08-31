import os
import handlePhs as hp
import numpy as np
import plotSig as ps
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from skimage.feature import graycomatrix, graycoprops
from rich.progress import track

def to255(array,length,width):

    ymax = 255
    ymin = 0
    xmax = max(map(max,array))
    xmin = min(map(min,array))
    
    for i in range(length):
        for j in range(width):
            array[i][j] = round(((ymax-ymin)*(array[i][j]-xmin)/(xmax-xmin))+ymin)
    
    return array

fileList='./filelist1'
headerFile='./data1/20180102-20180108.phs.ovr.rsc'
result='./result1'
sumOfRatio=0.85

header, phsStack, fileNum=hp.readFile(fileList,headerFile)

pca = PCA()
pca.fit(phsStack)
ratio=pca.explained_variance_ratio_

sum=0
for i in range(len(ratio)):
    sum+=ratio[i]
    if(sum>sumOfRatio):
        print(i)
        print(sum)
        break

numOfPCA=i

phsMean=phsStack.mean(axis=0)
phsSubMeanStack=phsStack-phsMean
# Compute ICA
ica = FastICA(n_components=numOfPCA, whiten='unit-variance')
S_ = ica.fit_transform(phsSubMeanStack)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
S_ = S_.astype('float32')
A_ = A_.astype('float32')

print(S_.shape)
print(A_.shape)

contrast=np.empty(S_.shape[1])
dissimilarity=np.empty(S_.shape[1])
energy=np.empty(S_.shape[1])
correlation=np.empty(S_.shape[1])
homogeneity=np.empty(S_.shape[1])

for i in range(S_.shape[1]):
    a=S_[:,i]
    b=np.round(a.reshape([header.length,header.width])).astype('int64')
    gray_image=to255(b,header.length,header.width)

    d = [5]  # 距离
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 角度

    # 计算GLCM
    glcm = graycomatrix(gray_image, distances=d, angles=angles, levels=256, symmetric=True, normed=True)

    # 计算GLCM特征
    contrast[i] = np.mean(graycoprops(glcm, 'contrast'))
    dissimilarity[i] = np.mean(graycoprops(glcm, 'dissimilarity'))
    energy[i] = np.mean(graycoprops(glcm, 'energy'))
    correlation[i] = np.mean(graycoprops(glcm, 'correlation'))
    homogeneity[i] = np.mean(graycoprops(glcm, 'homogeneity'))

    print(i)
    # 输出特征值
    print("Contrast:", contrast[i])
    print("Dissimilarity:", dissimilarity[i])
    print("Energy:", energy[i])
    print("Correlation:", correlation[i])
    print("Homogeneity:", homogeneity[i])

# print(np.argmax(contrast))
# print(np.argmax(dissimilarity))
# print(np.argmin(energy))
# print(np.argmin(correlation))
# print(np.argmax(homogeneity))

indexOfGLCM = np.empty([3])
indexOfGLCM[0] = np.argmax(contrast)
indexOfGLCM[1] = np.argmax(dissimilarity)
indexOfGLCM[2] = np.argmin(energy)
indexOfGLCM = indexOfGLCM.astype('int64')
print(indexOfGLCM)
countsOfIndex = np.bincount(indexOfGLCM)
print(countsOfIndex)
indexToDelete = np.argmax(countsOfIndex)
print(indexToDelete.shape)
# if (indexToDelete.shape[0] > 1):
#     print("GLCM failed to find a col to delete")
#     inputToDelete = int(input("input the index to delete"))
#     if (inputToDelete == -1 ):
#         newS = S_
#         newA = A_
#     elif (inputToDelete < -1 or inputToDelete > S_.shape[1]):
#         print("error input and exit")
#         exit()
#     elif (inputToDelete >=0 and inputToDelete <= S_.shape[1]):
#         indexToDelete = inputToDelete
#         newS = np.delete(S_,obj=indexToDelete,axis=1)
#         newA = np.delete(A_,obj=indexToDelete,axis=1)
# else:
#     print(indexToDelete)
#     newS = np.delete(S_,obj=indexToDelete,axis=1)
#     newA = np.delete(A_,obj=indexToDelete,axis=1)

newS = np.delete(S_,obj=indexToDelete,axis=1)
newA = np.delete(A_,obj=indexToDelete,axis=1)

print(newS.shape)
print(newA.shape)

newPhsStack = np.dot(newS, newA.T) + ica.mean_.astype('float32')
print(newPhsStack.shape)

for i in track(range(S_.shape[1]), description="writing S files", total=S_.shape[1]):
    hp.writePhs(S_[:, i], result+"/S_"+str(i)+".sig")

for i in track(range(A_.shape[1]), description="writing A files", total=A_.shape[1]):
    hp.writePhs(A_[:, i], result+"/A_"+str(i)+".sig")

f = open(fileList, 'r')
i=0
for line in track(open(fileList), description="writing cor files", total=fileNum):
        line = f.readline()
        phs = hp.readPhs(line.strip())
        index = np.where(phs == 0)
        newPhsStack[index, i] = 0
        line = os.path.split(line)[1].strip()
        hp.writePhs(newPhsStack[:, i], result+"/"+line+".ICAcorr")
        i = i+1

#ps.saveSubFig(phsStack, header, result+'/input.png')
ps.saveSubFig(newPhsStack, header, result+'/output.png')
ps.saveSubFig(S_, header, result+'/S_all.png')
ps.saveLineFig(A_,result+'/A_all.png')
