# 大数据-矩阵分解

## 环境配置：
* Python3.7.4
* pytorch 1.7.1
* torchvision 0.8.2 + cu110
* Pandas


## 文件结构：
```
  ├── my_dataset.py: 读取数据，获得X_train X_test矩阵
  ├── task1.py: 协同过滤 -> 矩阵预测和损失计算全部使用矩阵运算
  ├── task1_slow.py: 协同过滤 -> weight矩阵使用矩阵运算，矩阵预测和损失使用遍历预测test中的非空值
  ├── task2.py: 矩阵分解 -> 使用nn.RMSE()
  ├── task2_plot.py 矩阵分解 -> 使用要求的损失函数，可出RMSE和目标函数值曲线图
```
 
## 数据集
* movie_titles.txt
* netflix_test.txt
* netflix_train.txt
* users.txt


  


