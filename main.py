from preprocessing import data_preprocessing
from preprocessing2 import value_preprocessing, rank_preprocessing
from modeling import modeling, model_MLP, model_CNN, model_RNN, model_RNN2, model_CNN_SVM, model_NGBoost, model_GRU, model_CatBoost
from drawing import drawing
import warnings
warnings.filterwarnings("ignore")


def main():
    data = data_preprocessing()
    # feature_1, label_1 = value_preprocessing(data[['improved_water', 'carbon_emissions', 'cultivated_land',
    #                                                'CRI_score', 'Fitness', 'is_BR', 'Continent',
    #                                                'Year', 'country_style']], index='country_style')
    # feature_2, label_2 = rank_preprocessing(data[['water_rank', 'carbon_rank', 'cultivated_rank', 'CRI_rank',
    #                                               'Fitness_rank', 'is_BR', 'Continent', 'Year', 'country_style']],
    #                                         index='country_style') FSI_rank2

    feature_1, label_1 = value_preprocessing(data, index='FSI_rank2')
    f1 = ['improved_water', 'carbon_emissions', 'cultivated_land',
          'Fitness']
    f2 = ['improved_water', 'cultivated_land',
          'CRI_score',  'is_BR', 'Continent']
    f3 = ['carbon_emissions',
          'CRI_score', 'Fitness', 'is_BR', 'Continent']
    f4 = ['CRI_score', 'Fitness', 'is_BR', 'Continent']
    f5 = ['improved_water', 'carbon_emissions',  'Fitness', 'CRI_score']
    f6 = ['carbon_emissions', 'cultivated_land',
          'is_BR', 'Continent']
    f7 = ['improved_water', 'carbon_emissions', 'cultivated_land',
          'CRI_score', 'Fitness']
    f8 = ['improved_water', 'carbon_emissions', 'cultivated_land',
          'CRI_score', 'Fitness', 'is_BR', 'Continent',
          'Year']

#     f_list = [f1, f2, f3, f4, f5, f6, f7, f8]

#     for i in f_list:
    #   model_CatBoost(feature_1[i], label_1)
    #   model_GRU(feature_1[i], label_1)
    #   model_NGBoost(feature_1[i], label_1)
    #   model_RNN(feature_1[i], label_1)
    #   model_CNN(feature_1[i], label_1)
#     model_MLP(feature_1[i], label_1)
    #     modeling(feature_1[i], label_1)
    #   modeling(feature_1[f8], label_1)
#     modeling(feature_1[f8], label_1)
    # model_CNN(feature_1[i], label_1)
    modeling(feature_1[f8], label_1)
#    model_CNN(feature_1[f8], label_1)
#    model_RNN(feature_1[f8], label_1)
#    model_GRU(feature_1[f8], label_1)
#    model_MLP(feature_1[f8], label_1)
#    model_NGBoost(feature_1[f8], label_1)
#    model_CNN(feature_1[f8], label_1)
#    model_CatBoost(feature_1[f8], label_1) # yes
#    model_CNN_SVM(feature_1[f8], label_1)
#    model_MLP(feature_1[f1], label_1) # yes
#    model_NN(feature_1, label_1)
#    model_xgboost(feature_1, label_1)
#    model_regr(data[['water_rank', 'carbon_rank', 'Fitness_rank']], data[['FSI_rank2']])
# drawing()


if __name__ == '__main__':
    main()
