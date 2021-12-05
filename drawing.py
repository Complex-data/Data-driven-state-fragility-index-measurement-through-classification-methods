import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# 绘制堆积柱状图
def stackedbarplot(x_data, y_data_list, y_data_names, colors, x_label, y_label, title):
    # plt.rcParams['figure.figsize'] = (13.66, 7.68)
    # plt.subplots_adjust(wspace=0, hspace=0)
    _, ax = plt.subplots()

    for i in range(0, len(y_data_list)):
        if i == 0:
            ax.bar(x_data, y_data_list[i],
                   color=colors[i],
                   align='center',
                   label=y_data_names[i],
                   alpha=0.5)
        else:
            ax.bar(x_data, y_data_list[i],
                   color=colors[i],
                   bottom=y_data_list[1 - i],
                   align='center',
                   label=y_data_names[i],
                   alpha=0.5)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(ncol=2, bbox_to_anchor=(0.7, 1.0), fontsize=8)
    ax.locator_params('x', nbins=5)
    # ax.spines['top'].set_visible(False)
    xs = ['DT(Entroy)', 'Random Forest', 'GBDT', 'XGBoost']
    plt.xticks(x_data, xs)
    plt.text(0.8, 20, '91%', family='serif', size=15, color='black', style='italic')
    plt.text(1.8, 27, '98%', family='serif', size=15, color='black', style='italic')
    plt.text(2.8, 35, '72%', family='serif', size=15, color='black', style='italic')
    plt.text(3.8, 40, '92%', family='serif', size=15, color='black', style='italic')
    plt.show()


def drawing():
    x_data = np.arange(1, 5, 1)
    y_data_list1 = pd.Series([91, 98, 72, 92])
    y_data_list2 = pd.Series([9, 2, 28, 8])
    y_data_list = list()
    y_data_list.append(y_data_list1)
    y_data_list.append(y_data_list2)
    y_data_name = ['Success', 'Fail']
    # # 调用绘图函数
    stackedbarplot(x_data=x_data
                   , y_data_list=y_data_list
                   , y_data_names=y_data_name
                   , colors=['red', 'blue']
                   , x_label=''
                   , y_label='Pearson Correlation Coefficient(%)'
                   , title='Pearson')
