import sys, os, math, time
import matplotlib.pyplot as plt
import numpy as np
import copy


def perceptron(w, x, l, lr):
    x1 = np.array([i for i in x])
    x1 = np.append(x1, [1])
    net = np.dot(w, x1)
    o = 1 if net >= 0 else -1
    w1 = np.array(w+lr*(l-o)*x1)
    return w1

def sgn(w,x):
    x1 = np.array([i for i in x])
    x1 = np.append(x1, [1])
    net = np.dot(w, x1)
    o = 1 if net >= 0 else -1
    return o

def flatten(dataset):
    fal = []
    num = len(dataset[0]) * len(dataset[0][0])
    for i in dataset:
        fal.append(np.reshape(i, num))
    return fal

def main():
    # original data
    C = np.array([[0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,0], [1,0,0,0,1], [0,1,1,1,0]])
    H = np.array([[1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1]])
    L = np.array([[1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,1,1,1,1]])

    # training set prepared
    num = len(C)*len(C[0])
    labels = [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]    # 数据格式[C类, H类, L类],每个元素是该类的类别编码[1,1,1,-1,-1]
    original = [C, H, L]
    trainSet_prepared = flatten(original)    # 数据格式[样本1，样本2，样本3],每个样本是一个长度为25的向量

    # 只有一个噪音的数据集,75
    # plt.figure()
    oneNoise = []
    label_oneNoise = [[], [], []]    # label_oneNoise=[[75], [75], [75]]
    for index, i in enumerate(original):
        fla_letter = flatten([i])[0]
        for index1, pix in enumerate(fla_letter):
            temp = copy.deepcopy(fla_letter)
            temp[index1] = 1 if pix == 0 else 0
            oneNoise.append(temp)
            for t in range(3):
                if t == index:  # C类
                    label_oneNoise[t].append(1)
                else:
                    label_oneNoise[t].append(-1)
            # if index1 < 6:
            #     plt.subplot(3,6,index*6+(index1+1))
            #     plt.imshow(np.reshape(temp, [5,5]),cmap = plt.cm.binary)
            #     plt.xticks([]), plt.yticks([])
    # plt.tight_layout()
    # plt.savefig('./task2_picture/one_noise.png')
    # plt.show()

    # 有两个噪点的数据集
    # plt.figure()
    testSet_twoNoise = []
    label_twoNoise = [[], [], []]  # label_twoNoise=[[75], [75], [75]]
    for index, i in enumerate(original):
        fla_letter = flatten([i])[0]
        for index1, pix1 in enumerate(fla_letter):
            for index2, pix2 in enumerate(fla_letter[index1 + 1:]):
                temp = copy.deepcopy(fla_letter)
                temp[index1] = 1 if pix1 == 0 else 0
                temp[index1+index2+1] = 1 if pix2 == 0 else 0
                testSet_twoNoise.append(temp)
                for t in range(3):
                    if t == index:  # C类
                        label_twoNoise[t].append(1)
                    else:
                        label_twoNoise[t].append(-1)
                # if index1 < 6 :
                #     plt.subplot(3, 6, index * 6 + (index1 + 1))
                #     plt.imshow(np.reshape(temp, [5, 5]), cmap=plt.cm.binary)
                #     plt.xticks([]), plt.yticks([])
    # plt.tight_layout()
    # plt.savefig('./task2_picture/two_noise.png')
    # plt.show()

    # training set
    labels_concate = [[], [], []]
    for index, i in enumerate(labels):
        labels_concate[index] = i + label_oneNoise[index]

    trainSet = trainSet_prepared + oneNoise # 合并原始字母和onenoise数据集

    # initialization
    # w = np.zeros(num+1)
    epochs = 20; lr = 1

    # train
    w_history = [[], [], []]    # C,H,L
    for letter in range(3):
        w = np.zeros(num+1)
        flag = 0
        for epoch in range(epochs):
            for x,l in zip(trainSet, labels_concate[letter]):
                w = perceptron(w,x,l,lr)
            w_history[letter].append(w)
            if epoch >= 1 and sum([np.square(i - j) for i, j in zip(w, w_history[letter][epoch - 1])]) == 0:
                flag = epoch + 1
                print(f'Training process stopped at epoch {flag}.')
                break


    # test in the testSet_twoNoise
    result = []
    for letter in range(3):
        w = w_history[letter][len(w_history[letter])-1]
        result_letter = []
        for sample in testSet_twoNoise:
            result_letter.append(sgn(w,sample))
        result.append(result_letter)
    # print(np.array(result))

    # 每个类的错分数量
    for letter in range(3):
        f = 0
        for index, i in enumerate(testSet_twoNoise):
            if result[letter][index] != label_twoNoise[letter][index]:
                f += 1
                print('This sample is misclassified.')
                # plt.figure()
                # plt.imshow(np.reshape(i, [5,5]))
                # plt.show()
        print(f)

    # visualization of w
    print(w_history)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(np.reshape(w_history[0][len(w_history[0])-1][:25], [5, 5]))
    plt.title('C', fontsize=15)
    plt.subplot(1, 3, 2)
    plt.imshow(np.reshape(w_history[1][len(w_history[1]) - 1][:25], [5, 5]))
    plt.title('H', fontsize=15)
    plt.subplot(1, 3, 3)
    plt.imshow(np.reshape(w_history[2][len(w_history[2]) - 1][:25], [5, 5]))
    plt.title('L', fontsize=15)
    plt.tight_layout()
    plt.savefig('./task2_picture/keshihua_2.png')
    plt.show()


if __name__ == '__main__':
    main()

