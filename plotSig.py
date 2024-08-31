import handlePhs as hp
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def saveFig(file_name, header_name, save_name):

    header = hp.readHeader(header_name)
    with open(file_name, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
        phs = np.reshape(data, [header.length, header.width])

    index = np.where(phs == 0)
    v_num = header.length*header.width-index[0].shape[0]
    v_mean = np.sum(phs)/v_num
    phs[index] = np.nan
    phs1 = phs-v_mean
    phs1 = np.square(phs1)
    phs1[index] = 0
    v_std = np.sqrt(np.sum(phs1)/v_num)

    xmin = header.xfirst
    xmax = header.xfirst+header.xstep*header.width
    ymax = header.yfirst
    ymin = header.yfirst+header.ystep*header.length

    plt.rc('font', family='Times New Roman', size=12)
    plt.imshow(phs, cmap='hsv', vmax=v_mean+3*v_std, vmin=v_mean -
               3*v_std, extent=[xmin, xmax, ymin, ymax])
    plt.colorbar(extend='both', fraction=0.02, pad=0.1)

    x_major_locator = MultipleLocator(0.5)
    y_major_locator = MultipleLocator(0.5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%3.1f°'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%3.1f°'))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # plt.show()
    plt.savefig(save_name, dpi=600)
    plt.close()


def saveSubFig(phsstack, header, savename):
    number = phsstack.shape[1]
    col = 6
    row = math.ceil(number/col)

    xmin = header.xfirst
    xmax = header.xfirst+header.xstep*header.width
    ymax = header.yfirst
    ymin = header.yfirst+header.ystep*header.length

    phss = phsstack
    nanindex = np.where(phsstack == 0)
    v_num = phss.shape[0]*phss.shape[1]-nanindex[0].shape[0]
    v_mean = np.sum(phss)/v_num
    phss[nanindex] = np.nan
    phs1 = phss-v_mean
    phs1 = np.square(phs1)
    phs1[nanindex] = 0
    v_std = np.sqrt(np.sum(phs1)/v_num)

    plt.figure(figsize=(14, 8))

    for i in range(0, number):
        id = i+1
        plt.subplot(row, col, id)

        data = phsstack[:, i]
        phs = np.reshape(data, [header.length, header.width])
        index = np.where(phs == 0)
        phs[index] = np.nan

        plt.imshow(phs, cmap='hsv', vmax=v_mean+3*v_std, vmin=v_mean -
                   3*v_std, extent=[xmin, xmax, ymin, ymax])
        x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(1)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%3.1f°'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%3.1f°'))

    plt.tight_layout()
    #plt.colorbar(extend='both', orientation='horizontal')
    # plt.show()
    plt.savefig(savename, dpi=600)
    plt.close()


def saveLineFig(A,savename):
    number = A.shape[1]
    rangeA = A.shape[0]
    fig, ax = plt.subplots(nrows=number, ncols=1, sharex=True)
    
    vmin=np.min(A)
    vmax=np.max(A)
    x_major_locator=MultipleLocator(5)
    #y_major_locator=MultipleLocator(1)
    for i in range(number):
        x = np.array(list(range(rangeA)))
        y = A[:, i]
        ax[i].plot(x, y)
        plt.rc('font',family='Times New Roman')
        #plt.ylim(vmin,vmax)
        ax1 = plt.gca()
        ax1.xaxis.set_major_locator(x_major_locator)
        #ax1.yaxis.set_major_locator(y_major_locator)
    #plt.show()
    plt.savefig(savename, dpi=600)
    plt.close()
