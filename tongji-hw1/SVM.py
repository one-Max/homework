from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ==========================================================
# divide df into X and y
def split_X_label(df):
    X = np.array(df.iloc[:, :2])
    y = np.array(df.iloc[:, 2])

    return X, y


def main():
    # data loader
    df = pd.read_csv('/data1/zjw/homework/tongji_hw1/Data.csv',
                     usecols=[0, 7, 21])
    df.rename(columns={'baseline value': 'base',
              'abnormal_short_term_variability': 'astv', 'fetal_health': 'label'}, inplace=True)

    # split into train and test set (0.7-0.3)
    train_data = df.sample(frac=0.7, random_state=0, axis=0)
    test_data = df[~df.index.isin(train_data.index)]

    # scaler = StandardScaler()
    # trainset preprocessing
    X_train, train_label = split_X_label(train_data)
    # X_train = scaler.fit_transform(X_train)

    # testset preprocessing
    X_test, test_label = split_X_label(test_data)
    # X_test = scaler.fit_transform(X_test)
    
    train_acc = []
    test_acc = []

    clf = svm.SVC(C=3, gamma=0.05, max_iter=200, kernel = 'rbf')
    clf.fit(X_train, train_label)

    #Test on Training data
    train_result = clf.predict(X_train)
    acc_tr = sum(train_result == train_label)/train_label.shape[0]
    print('Training accuracy: ', f'{acc_tr:.6f}')
    train_acc.append(acc_tr)

    #Test on test data
    test_result = clf.predict(X_test)
    acc_te = sum(test_result == test_label)/test_label.shape[0]
    print('    Test accuracy: ', f'{acc_te:.6f}')
    test_acc.append(acc_te)


if __name__ == '__main__':
    main()