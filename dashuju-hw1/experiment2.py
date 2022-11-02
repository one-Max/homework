import sys, os, math, time, copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import NAN
from scipy import stats


def draw_and_test(series, *args, column=None):
    # 画一个series的统计直方图，密度图，并判断是否符合正态分布
    s = pd.DataFrame(np.array(series), columns=[column])

    fig = plt.figure(figsize=(10, 6), dpi=150)
    ax1 = fig.add_subplot(1, 1, 1)
    s.hist(bins=20, alpha=0.5, ax=ax1, rwidth=0.9)              # 绘制直方图
    s.plot(kind='kde', secondary_y=True, ax=ax1, legend=None)   # 绘制密度图，使用双坐标轴

    ax1.grid()
    ax1.tick_params(labelsize=10)
    font = {'family': 'Times New Roman'}
    ax1.set_ylabel('Density', fontdict=font, fontsize=15)

    if len(args) != 0:
        ax1.set_title(f'Probability density function of category {args[0]}', fontdict=font, fontsize=18)
        ax1.set_xlabel(f'{column}', fontdict=font, fontsize=15)
        plt.tight_layout()
        plt.savefig(f'{column}_pdf_{args[0]}.png')
    else:
        ax1.set_title(f'Probability density function of all sample', fontdict=font, fontsize=18)
        ax1.set_xlabel(f'{column}', fontdict=font, fontsize=15)
        plt.tight_layout()
        plt.savefig(f'{column}_pdf_all.png')
    plt.show()

    # 计算均值,标准差
    u = s[column].mean()
    std = s[column].std()

    if len(args) != 0:
        print(f'第{args[0]}类的 {column} kstest统计检验结果:--------------------------------------------')
        print(stats.kstest(s[column], 'norm', (u, std)))
        print()
    else:
        print(f'全部样本的 {column} kstest统计检验结果:-------------------------------------------------')
        print(stats.kstest(s[column], 'norm', (u, std)))
        print()


def statistics(data):
    # series是pandas df的一列，对某一列做统计分析
    column = data.columns.values[1]
    cat_num = data['群类别'].max()
    mean_all = data[column].mean()
    group = []      # 5个list，包含属于类i的所有群
    mean = []       # 5个int，包含每个类的均值
    num = []        # 每个类的sample数量
    var = []        # 每个类的ss_w

    for category in range(cat_num):
        group.append([])
        num_fl = 0; sum = 0
        for i in range(len(data[column])):
            if data['群类别'][i] == category + 1:
                group[category].append(data[column][i])
                num_fl += 1
                sum += data[column][i]
        num.append(num_fl)
        mean.append(sum / num_fl)

    for category in range(cat_num):
        squared_sum = 0
        for i in group[category]:
            squared_sum += np.square(i - mean[category])
        var.append(squared_sum)

    std = [np.sqrt(i / (num[index] - 1)) for index, i in enumerate(var)]

    print(f'The mean of all samples : {mean_all}')
    print(f'The mean of these classes : {mean}')
    print(f'The samples of these classes : {num}')
    print(f'The variance of these classes : {var}')
    print(f'The std of these classes : {std}')
    print()
    dic = {'group':group,
           'mean':mean,
           'var':var,
           'num':num,
           'mean_all':mean_all,
           'std':std
           }
    return dic


def one_way_ANOVA(collect):
    # 单因素方差分析
    ssw = np.sum(collect['var'])
    ssb = 0
    for index, mean in enumerate(collect['mean']):
        ssb += collect['num'][index] * np.square(mean - collect['mean_all'])
    print(f'SSw : {ssw}; SSb : {ssb}')

    df_b = len(collect['group']) - 1
    df_w = np.sum(collect['num']) - len(collect['group'])
    df = df_w + df_b
    print(f'df_b : {df_b}; df_w : {df_w}; df : {df}')

    MS_b = ssb/df_b; MS_w = ssw/df_w
    print(f'MS_b : {MS_b}; MS_w : {MS_w}')

    F = MS_b/MS_w
    print(f'F : {F}')

    # 调库检验正确性：单因素方差分析f_oneway
    print('f_oneway统计检验结果:----------------------------------------------------')
    print(stats.f_oneway(*collect['group'], axis=0))
    print()


def main():
    dataset = pd.read_excel('data.xlsx', index_col=None)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 1000)
    pd.set_option("display.max_colwidth", 50)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    # print(data)

    # print(data.isnull().sum()) # 缺省值，群名中有一个
    dataset['群名'].fillna('no name', inplace = True)
    # print(data.describe())

    dic_match = {'ch':['平均年龄', '性别比', '地域集中度', '会话数'],
                 'en':['Average age', 'Sex ratio', 'Geographical concentration', 'Number of sessions']}

    # # 以下取消注释得到 original
    # for col_ch, col_en in zip(dic_match['ch'], dic_match['en']):
    #     data = dataset[['群类别', col_ch]].copy()
    #
    #     # 获得某一列的统计数据：对该列数据分组，均值，var，num，std，全体均值
    #     collect = statistics(data)
    #     # 对某一列做单因素方差分析
    #     one_way_ANOVA(collect)
    #     # pdf绘图以及正态分布的检验
    #     draw_and_test(data[col_ch], column=col_en)
    #     for i in range(data['群类别'].max()):
    #         draw_and_test(collect['group'][i], i+1, column=col_en)

    # 以下取消注释得到 log_transformation
    for col_ch, col_en in zip(dic_match['ch'][1:], dic_match['en'][1:]):
        data = dataset[['群类别', col_ch]].copy()
        # 由于log(0)不存在，对所有数据加0.001后取对数
        data.loc[:, col_ch] = np.log(data.loc[:, col_ch] + 10e-4)
        collect = statistics(data)
        one_way_ANOVA(collect)
        draw_and_test(data[col_ch], column='log_' + col_en)
        for i in range(data['群类别'].max()):
            draw_and_test(collect['group'][i], i + 1, column='log_' + col_en)


if __name__ == '__main__':
    main()

