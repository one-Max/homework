import os
import time
import copy
import matplotlib.pyplot as plt
import numpy as np

# ==================================================
# 生成只有1个噪音的数据集,75
def gerenate_one_noise(data):
    # plt.figure()
    oneNoise = []
    for index, i in enumerate(data):
        for index1, pix in enumerate(i):
            temp = copy.deepcopy(i)
            temp[index1] = 1 if pix == 0 else 0
            oneNoise.append(temp)
            # if index1 < 6:
            #     plt.subplot(3,6,index*6+(index1+1))
            #     plt.imshow(np.reshape(temp, [5,5]),cmap = plt.cm.binary)
            #     plt.xticks([]), plt.yticks([])
    # plt.tight_layout()
    # plt.savefig('./one_noise.png')
    # plt.show()
    return oneNoise

# ==================================================
# 生成有2个噪音的数据集
def gerenate_two_noise(data):
    # plt.figure()
    twoNoise = []
    for index, i in enumerate(data):
        for index1, pix1 in enumerate(i):
            for index2, pix2 in enumerate(i[index1 + 1:]):
                temp = copy.deepcopy(i)
                temp[index1] = 1 if pix1 == 0 else 0
                temp[index1+index2+1] = 1 if pix2 == 0 else 0
                twoNoise.append(temp)
                # if index1 < 6 :
                #     plt.subplot(3, 6, index * 6 + (index1 + 1))
                #     plt.imshow(np.reshape(temp, [5, 5]), cmap=plt.cm.binary)
                #     plt.xticks([]), plt.yticks([])
    # plt.tight_layout()
    # plt.savefig('./two_noise.png')
    # plt.show()
    return twoNoise

def main():
    # original data
    C = [0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0]
    H = [1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1]
    L = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1]

    # training set prepared
    X_origin = np.array([C, H, L]).astype('float32')
    
    # generate noise dataset
    data_onenoise = gerenate_one_noise(X_origin)
    data_twonoise = gerenate_two_noise(X_origin)

    # 保存数据
    np.save('homework/ann-hw3/data/data_onenoise.npy', data_onenoise)
    np.save('homework/ann-hw3/data/data_twonoise.npy', data_twonoise)


if __name__ == "__main__":
    main()
