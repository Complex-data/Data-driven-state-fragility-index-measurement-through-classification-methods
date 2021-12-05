# feature preprocessing
# ...... #
from sklearn.preprocessing import MinMaxScaler, StandardScaler        # 归一化和标准化函数
from sklearn.preprocessing import LabelEncoder, OneHotEncoder         # 数值化-标签化和独热编码函数
from sklearn.preprocessing import Normalizer                         # 正规化函数
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # LDA降维函数
from sklearn.decomposition import PCA
import os


# fr:FSI_rank---False:MinMaxScaler;True:StandardScaler
# iw:improved_water---False:MinMaxScaler;True:StandardScaler
# ce:carbon_emissions---False:MinMaxScaler;True:StandardScaler
# cl;cultivated_land---False:MinMaxScaler;True:StandardScaler
# cs:CRI_score---False:MinMaxScaler;True:StandardScaler
# fn:Fitness---False:MinMaxScaler;True:StandardScaler
# ibr:is_BR---False:LabelEncoding;True:OneHotEncoding
# cn:Continent---False:LabelEncoding;True:OneHotEncoding
# cse:country_style---False:MinMaxScaler;True:StandardScaler
# yr:Year---False:LabelEncoding;True:OneHotEncoding
# lower_d:lower dimension---Fales:keep dimension;Ture:dimensionality reduction
def value_preprocessing(data, index, fr=False, iw=False, ce=False, cl=False, cs=False, fn=False,
                        ibr=False, cn=False, yr=False, cse=False, lower_d=False, ld_n=1):
    df = data.copy()
    print(df.index)

    # 1、清洗数据
    df.dropna()

    # 2、得到标注，因为标注不确定，统一进行归一化处理或标准化处理
    # if not fr:
    #     df.loc[:, [index]] = MinMaxScaler().fit_transform(df.loc[:, [index]].values.reshape(-1, 1)).reshape(1, -1)[0]
    # else:
    #     df.loc[:, [index]] = StandardScaler().fit_transform(df.loc[:, [index]].values.reshape(-1, 1)).reshape(1, -1)[0]
    # label = df.loc[:, [index]]
    # df = df.drop(index, axis=1)
    # 1、得到标注
    label = df[index]
    df = df.drop(index, axis=1)

    # 3、特征选择
    # 参照上述pearson相关系数和spearman相关系数
    # 当特征较少，尽量保留全特征

    # 4、特征处理
    # 连续数值归一化或者标准化
    scaler_lst = [iw, ce, cl, cs, fn]
    column_lst = ['improved_water', 'carbon_emissions',
                  'cultivated_land', 'CRI_score', 'Fitness', 'country_style']
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            df[column_lst[i]] = MinMaxScaler().fit_transform(
                df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            df[column_lst[i]] = StandardScaler().fit_transform(
                df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]

            # 标签化或者独热编码
    scaler_lst = [ibr, cn, yr]
    column_lst = ['is_BR', 'Continent', 'Year']
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            if column_lst[i] == 'is_BR':
                df[column_lst[i]] = [map_ibr(s) for s in df['is_BR'].values]
            elif column_lst[i] == 'Year':
                df[column_lst[i]] = [map_yr(s) for s in df['Year'].values]
            else:
                df[column_lst[i]] = LabelEncoder(
                ).fit_transform(df[column_lst[i]])
            # 标签化之后归一化
            df[column_lst[i]] = MinMaxScaler().fit_transform(
                df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            df = pd.get_dummies(df, columns=[column_lst[i]])

    # 降维
    # 因为LDA降维只保留一个特征 因此使用PCA降维
    if lower_d:
        return PCA(n_components=ld_n).fit_transform(df.values), label
    return df, label


d_ibr = dict([('no', 0), ('is', 1)])
d_yr = dict([(2010, 0), (2011, 1), (2012, 2), (2013, 3), (2014, 4)])


def map_ibr(s):
    return d_ibr.get(s, 0)


def map_yr(s):
    return d_yr.get(s, 0)


# fr2:FSI_rank2---False:MinMaxScaler;True:StandardScaler
# wr:water_rank---False:MinMaxScaler;True:StandardScaler
# cbr:carbon_rank---False:MinMaxScaler;True:StandardScaler
# ctr;cultivated_rank---False:MinMaxScaler;True:StandardScaler
# crr:CRI_rank---False:MinMaxScaler;True:StandardScaler
# fnr:Fitness_rank---False:MinMaxScaler;True:StandardScaler
# ibr:is_BR---False:LabelEncoding;True:OneHotEncoding
# cn:Continent---False:LabelEncoding;True:OneHotEncoding
# yr:Year---False:LabelEncoding;True:OneHotEncoding
# lower_d:lower dimension---Fales:keep dimension;Ture:dimensionality reduction
def rank_preprocessing(data, index, fr2=False, wr=False, cbr=False, ctr=False, crr=False, fnr=False, ibr=False,
                       cn=False, yr=False, cse=False, lower_d=False, ld_n=1):
    df = data.copy()

    # 1、清洗数据
    df.dropna()

    # 2、得到标注，因为'FSI_rank'的值覆盖范围较大 首先进行归一化或标准化
    # if not fr2:
    #     df['FSI_rank2'] = MinMaxScaler().fit_transform(df['FSI_rank2'].values.reshape(-1, 1)).reshape(1, -1)[0]
    # else:
    #     df['FSI_rank2'] = StandardScaler().fit_transform(df['FSI_rank2'].values.reshape(-1, 1)).reshape(1, -1)[0]
    # label = df['FSI_rank2']
    # df = df.drop('FSI_rank2', axis=1)
    label = df[index]
    df = df.drop(index, axis=1)

    # 3、特征选择
    # 参照上述pearson相关系数和spearman相关系数
    # 当特征较少，尽量保留全特征

    # 4、特征处理
    # 连续数值归一化或者标准化
    scaler_lst = [wr, cbr, ctr, crr, fnr]
    column_lst = ['water_rank', 'carbon_rank',
                  'cultivated_rank', 'CRI_rank', 'Fitness_rank']
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            df[column_lst[i]] = MinMaxScaler().fit_transform(
                df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            df[column_lst[i]] = StandardScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[
                0]

            # 标签化或者独热编码
    scaler_lst = [ibr, cn, yr]
    column_lst = ['is_BR', 'Continent', 'Year']
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            if column_lst[i] == 'is_BR':
                df[column_lst[i]] = [map_ibr(s) for s in df['is_BR'].values]
            elif column_lst[i] == 'Year':
                df[column_lst[i]] = [map_yr(s) for s in df['Year'].values]
            else:
                df[column_lst[i]] = LabelEncoder(
                ).fit_transform(df[column_lst[i]])
            # 标签化之后归一化
            df[column_lst[i]] = MinMaxScaler().fit_transform(
                df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            df = pd.get_dummies(df, columns=[column_lst[i]])

    # 降维
    # 因为LDA降维只保留一个特征 因此使用PCA降维
    if lower_d:
        return PCA(n_components=ld_n).fit_transform(df.values), label
    return df, label
