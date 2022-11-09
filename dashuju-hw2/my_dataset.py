import numpy as np
import pandas as pd

# ==================================================
# 还原出训练和测试用的矩阵
def data_loader(dataset, user_dict):
    X = np.zeros((10000, 10000))
    for index, data_row in enumerate(dataset):
        percent = int(100*index/len(dataset))
        print('\r'+'▇'*(percent//2)+str(percent)+'%', end='')
        
        data = data_row.split(' ')
        user_id = user_dict[int(data[0])]
        movie_id = int(data[1])-1
        score = int(data[2])
        X[user_id][movie_id] = score

    return X


def main():
    # ==================================================
    # 读取需要的三个文件
    data_root = 'homework/dashuju-hw2/data/'
    user_title = []
    data_train = []
    data_test = []

    with open(data_root+'users.txt','r',encoding='UTF-8') as f:
        for line in f.readlines():
            user_title.append(int(line.strip()))
    with open(data_root+'netflix_train.txt','r',encoding='UTF-8') as f:
        for line in f.readlines():
            data_train.append(line.strip())
    with open(data_root+'netflix_test.txt', 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            data_test.append(line.strip())

    # ==================================================
    # 生成训练和测试用的两个10000x10000的矩阵
    user_index = np.arange(len(user_title))
    user_dict = dict(zip(user_title, user_index))

    X_train = data_loader(data_train, user_dict)
    X_test = data_loader(data_test, user_dict)

    # ==================================================
    # 保存数据
    np.save('x_train1.npy', X_train)
    np.save('x_test1.npy', X_test)


if __name__ == "__main__":
    main()