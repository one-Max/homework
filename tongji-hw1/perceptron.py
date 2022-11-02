from symbol import parameters
import sys, os, math, time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ==========================================================
# divide df into X and y
def split_X_label(df):
    X = np.array(df.iloc[:, :2])
    y = np.array(df.iloc[:, 2])
    label = [[], [], []]
    for i in y:
        for t in range(0,3):
            if i == t+1:  
                label[t].append(1)
            else:
                label[t].append(-1)

    return X, np.array(label).T


# ==========================================================
# get augmented metrix of X
def extend(X):
    X_e = []
    for row in X:
        x1 = np.array([i for i in row])
        x1 = np.append(x1, [1])
        X_e.append(x1)
        
    return np.array(X_e)


# ==========================================================
def perceptron(w, x, l, lr):
    net = np.dot(w, x)
    o = 1 if net >= 0 else -1
    w1 = np.array(w+lr*(l-o)*x)
    return w1


# ==========================================================
# train 
def train(X, label, w, lr):

    num_class = len(label[0])

    # forward propagation
    label_pred = np.dot(X, w.T)

    # calculate accuracy
    acc = evaluate(label, label_pred)

    # backward propogation % update weight
    for x, l in zip(X, label):
        for c in range(num_class):
            w[c] = perceptron(w[c], x, l[c], lr)
    
    return w, acc


# ==========================================================
# calculate accu
def evaluate(label, label_pred):
    result = []
    k = 0
    for true, pred in zip(label, label_pred):
        flag = 0
        for i in true * pred:
            if i < 0:
                flag += 1
        if flag > 0:
            result.append(0)
        else:
            result.append(1)
            k += 1

    return k / len(result)


def main():

    # data loader
    df = pd.read_csv('/data1/zjw/homework/tongji_hw1/Data.csv',
                     usecols=[0, 7, 21])
    df.rename(columns={'baseline value': 'base', 'abnormal_short_term_variability': 'astv', 'fetal_health':'label'}, inplace = True)

    
    
    # split into train and test set (0.7-0.3)
   
    train_data = df.sample(frac=0.7, random_state=0, axis=0)
    test_data = df[~df.index.isin(train_data.index)]

    # trainset preprocessing
    X, label = split_X_label(train_data)
    X = (X-X.mean(axis=0))/X.std(axis=0)
    X_e = extend(X)
    
    # testset preprocessing
    X_test, label_test = split_X_label(test_data)
    X_test = (X_test-X_test.mean(axis=0))/X_test.std(axis=0)
    X_e_test = extend(X_test)

    np.random.seed(0)

    # model initialization
    w = np.random.random([3, 3])

    # train
    num_epochs = 15
    for epoch in range(num_epochs):
        w, acc = train(X_e, label, w, 0.4)

    # result
    label_pred = np.dot(X_e, w.T)
    evaluate(label, label_pred)
    print(f'acc_train: {acc:.6f}')

    # test
    label_pred_test = np.dot(X_e_test, w.T)
    acc_test = evaluate(label_test, label_pred_test)
    print(f'acc_test : {acc_test:.6f}')

    

if __name__ == '__main__':
    main()
