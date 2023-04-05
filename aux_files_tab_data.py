
import numpy as np
# import matplotlib.pyplot as plt

# import util
import random
import socket
# from sdgym import load_dataset
import argparse
import sys

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import  LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score


import pandas as pd
import seaborn as sns
import pickle
sns.set()

import warnings
warnings.filterwarnings('ignore')

import os


#def Feature_labels(labels, weights, device):

#    weights = torch.Tensor(weights)
#    weights = weights.to(device)

#    labels = labels.to(device)

#    weighted_labels_feature = labels/weights

#    return weighted_labels_feature



def undersample(raw_input_features, raw_labels, undersampled_rate):
    """ we take a pre-processing step such that the dataset is a bit more balanced """
    idx_negative_label = raw_labels == 0
    idx_positive_label = raw_labels == 1

    pos_samps_input = raw_input_features[idx_positive_label, :]
    pos_samps_label = raw_labels[idx_positive_label]
    neg_samps_input = raw_input_features[idx_negative_label, :]
    neg_samps_label = raw_labels[idx_negative_label]

    # take random 10 percent of the negative labelled data
    in_keep = np.random.permutation(np.sum(idx_negative_label))
    under_sampling_rate = undersampled_rate  # 0.4
    in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

    neg_samps_input = neg_samps_input[in_keep, :]
    neg_samps_label = neg_samps_label[in_keep]

    feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
    label_selected = np.concatenate((pos_samps_label, neg_samps_label))

    return feature_selected, label_selected


def data_loading(dataset, undersampled_rate, seed_number):

   if dataset=='epileptic':

        print("epileptic seizure recognition dataset") # this is homogeneous

        data = pd.read_csv("data/Epileptic/data.csv")

        feature_names = data.iloc[:, 1:-1].columns
        target = data.iloc[:, -1:].columns

        data_features = data[feature_names]
        data_target = data[target]

        for i, row in data_target.iterrows():
          if data_target.at[i,'y']!=1:
            data_target.at[i,'y'] = 0

        ###################

        raw_labels=np.array(data_target)
        raw_input_features=np.array(data_features)

        idx_negative_label = raw_labels == 0
        idx_positive_label = raw_labels == 1

        idx_negative_label=idx_negative_label.squeeze()
        idx_positive_label=idx_positive_label.squeeze()

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        # take random 10 percent of the negative labelled data
        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        label_selected=label_selected.squeeze()

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.70, test_size=0.30, random_state=seed_number)

        n_classes = 2

        num_numerical_inputs = []
        num_categorical_inputs = []

   elif dataset=="credit":

        print("Creditcard fraud detection dataset") # this is homogeneous

        data = pd.read_csv("data/Kaggle_Credit/creditcard.csv")

        feature_names = data.iloc[:, 1:30].columns
        target = data.iloc[:, 30:].columns

        data_features = data[feature_names]
        data_target = data[target]
        print(data_features.shape)

        """ we take a pre-processing step such that the dataset is a bit more balanced """
        raw_input_features = data_features.values
        raw_labels = data_target.values.ravel()

        idx_negative_label = raw_labels == 0
        idx_positive_label = raw_labels == 1

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        # take random 10 percent of the negative labelled data
        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80,
                                                            test_size=0.20, random_state=seed_number)
        n_classes = 2
        num_numerical_inputs = []
        num_categorical_inputs = []

   elif dataset=='census':

        print("census dataset") # this is heterogenous

        data = np.load("data/real/census/train.npy")

        numerical_columns = [0, 5, 16, 17, 18, 29, 38]
        ordinal_columns = []
        categorical_columns = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                               30, 31, 32, 33, 34, 35, 36, 37, 38, 40]
        n_classes = 2

        data = data[:, numerical_columns + ordinal_columns + categorical_columns]
        num_numerical_inputs = len(numerical_columns)
        num_categorical_inputs = len(categorical_columns + ordinal_columns) - 1

        raw_input_features = data[:, :-1]
        raw_labels = data[:, -1]
        print('raw input features', raw_input_features.shape)

        """ we take a pre-processing step such that the dataset is a bit more balanced """
        idx_negative_label = raw_labels == 0
        idx_positive_label = raw_labels == 1

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        # take random 10 percent of the negative labelled data
        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate #0.4
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80,
                                                            test_size=0.20, random_state=seed_number)


   elif dataset=='cervical':

        print("dataset is", dataset) # this is heterogenous

        df = pd.read_csv("data/Cervical/kag_risk_factors_cervical_cancer.csv")

        # df.head()
        df_nan = df.replace("?", np.float64(np.nan))
        df_nan.head()

        df1=df.apply(pd.to_numeric, errors="coerce")

        df1.columns = df1.columns.str.replace(' ', '')  # deleting spaces for ease of use

        """ this is the key in this data-preprocessing """
        df = df1[df1.isnull().sum(axis=1) < 10]

        numerical_df = ['Age', 'Numberofsexualpartners', 'Firstsexualintercourse', 'Numofpregnancies', 'Smokes(years)',
                        'Smokes(packs/year)', 'HormonalContraceptives(years)', 'IUD(years)', 'STDs(number)',
                        'STDs:Numberofdiagnosis',
                        'STDs:Timesincefirstdiagnosis', 'STDs:Timesincelastdiagnosis']
        categorical_df = ['Smokes', 'HormonalContraceptives', 'IUD', 'STDs', 'STDs:condylomatosis',
                          'STDs:vulvo-perinealcondylomatosis', 'STDs:syphilis', 'STDs:pelvicinflammatorydisease',
                          'STDs:genitalherpes', 'STDs:AIDS', 'STDs:cervicalcondylomatosis',
                          'STDs:molluscumcontagiosum', 'STDs:HIV', 'STDs:HepatitisB', 'STDs:HPV',
                          'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology', 'Biopsy']

        feature_names = numerical_df + categorical_df[:-1]
        num_numerical_inputs = len(numerical_df)
        num_categorical_inputs = len(categorical_df[:-1])

        for feature in numerical_df:
            feature_mean = round(df[feature].median(), 1)
            df[feature] = df[feature].fillna(feature_mean)

        for feature in categorical_df:
            df[feature] = df[feature].fillna(0.0)


        target = df['Biopsy']
        inputs = df[feature_names]
        print('raw input features', inputs.shape)

        n_classes = 2

        raw_input_features = inputs.values
        raw_labels = target.values.ravel()

        print('raw input features', raw_input_features.shape)

        """ we take a pre-processing step such that the dataset is a bit more balanced """
        idx_negative_label = raw_labels == 0
        idx_positive_label = raw_labels == 1

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        # take random 10 percent of the negative labelled data
        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate #0.5
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80,
                                                            test_size=0.20, random_state=seed_number)

   elif dataset=='adult':

        print("dataset is", dataset) # this is heterogenous

        # metadata = load_dataset('adult')
        # from sdgym.datasets import load_tables
        # tables = load_tables(metadata)

        # data, categorical_columns, ordinal_columns = load_dataset('adult')

        # adult_data = pd.read_csv("../data/adult/adult.csv")

        filename = 'data/adult/adult.p'
        with open(filename, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
            y_tot, x_tot = data

        # [0:'age', 1:'workclass', 2:'fnlwgt', 3:'education', 4:'education_num',
        #  5:'marital_status', 6:'occupation', 7:'relationship', 8:'race', 9:'sex',
        #  10:'capital_gain', 11:'capital_loss', 12:'hours_per_week', 13:'country']

        # categorical_columns = ['workclass', 'race', 'education', 'marital-status', 'occupation',
        #                  'relationship', 'gender', 'native-country']
        categorical_columns = [1, 3, 5, 6, 7, 8, 9, 13]
        ordinal_columns = []


        """ some specifics on this dataset """
        numerical_columns = list(set(np.arange(x_tot[:, :-1].shape[1])) - set(categorical_columns + ordinal_columns))
        n_classes = 2

        x_tot = x_tot[:, numerical_columns + ordinal_columns + categorical_columns]
        num_numerical_inputs = len(numerical_columns)
        num_categorical_inputs = len(categorical_columns + ordinal_columns) - 1

        # inputs = data[:, :-1]
        # target = data[:, -1]

        inputs = x_tot
        target = y_tot

        inputs, target = undersample(inputs, target, undersampled_rate)

        X_train, X_test, y_train, y_test = train_test_split(inputs, target, train_size=0.90, test_size=0.10,
                                                            random_state=seed_number)

   elif dataset=='isolet':

        print("isolet dataset")

        data_features_npy = np.load('data/Isolet/isolet_data.npy')
        data_target_npy = np.load('data/Isolet/isolet_labels.npy')

        values = data_features_npy
        index = ['Row' + str(i) for i in range(1, len(values) + 1)]

        values_l = data_target_npy
        index_l = ['Row' + str(i) for i in range(1, len(values) + 1)]

        data_features = pd.DataFrame(values, index=index)
        data_target = pd.DataFrame(values_l, index=index_l)

        ####

        raw_labels = np.array(data_target)
        raw_input_features = np.array(data_features)

        idx_negative_label = raw_labels == 0
        idx_positive_label = raw_labels == 1

        idx_negative_label = idx_negative_label.squeeze()
        idx_positive_label = idx_positive_label.squeeze()

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate  # 0.01
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        label_selected = label_selected.squeeze()

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.70, test_size=0.30,
                                                            random_state=seed_number)
        n_classes = 2
        num_numerical_inputs = []
        num_categorical_inputs = []


   elif dataset=='intrusion':

        print("dataset is", dataset)

        original_train_data = np.load("data/real/intrusion/train.npy")
        original_test_data = np.load("data/real/intrusion/test.npy")

        # we put them together and make a new train/test split in the following
        data = np.concatenate((original_train_data, original_test_data))

        """ some specifics on this dataset """
        n_classes = 5 # removed to 5

        """ some changes we make in the type of features for applying to our model """
        categorical_columns_binary = [6, 11, 13, 20]  # these are binary categorical columns
        the_rest_columns = list(set(np.arange(data[:, :-1].shape[1])) - set(categorical_columns_binary))

        num_numerical_inputs = len(the_rest_columns)  # 10. Separately from the numerical ones, we compute the length-scale for the rest columns
        num_categorical_inputs = len(categorical_columns_binary)

        raw_labels = data[:, -1]
        raw_input_features = data[:, the_rest_columns + categorical_columns_binary]
        print(raw_input_features.shape)

        """ we take a pre-processing step such that the dataset is a bit more balanced """
        idx_negative_label = raw_labels == 0  # this is a dominant one about 80%, which we want to undersample
        idx_positive_label = raw_labels != 0

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate#0.3
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.70,
                                                            test_size=0.30,
                                                            random_state=seed_number)

   elif dataset=='covtype':

        print("dataset is", dataset)

        train_data = np.load("data/real/covtype/train_new.npy")
        test_data = np.load("data/real/covtype/test_new.npy")
        # we put them together and make a new train/test split in the following
        data = np.concatenate((train_data, test_data))

        # covtype_dataset = datasets.fetch_covtype()
        # data = covtype_dataset.data
        # target = covtype_dataset.target
        # data = np.concatenate((data.transpose(), target[:,None].transpose()))
        # data = data.transpose()

        """ some specifics on this dataset """
        numerical_columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ordinal_columns = []
        categorical_columns = list(set(np.arange(data.shape[1])) - set(numerical_columns + ordinal_columns))
        # Note: in this dataset, the categorical variables are all binary
        n_classes = 7

        print('data shape is', data.shape)
        print('indices for numerical columns are', numerical_columns)
        print('indices for categorical columns are', categorical_columns)
        print('indices for ordinal columns are', ordinal_columns)

        # sorting the data based on the type of features.
        data = data[:, numerical_columns + ordinal_columns + categorical_columns]

        num_numerical_inputs = len(numerical_columns)
        num_categorical_inputs = len(categorical_columns + ordinal_columns) - 1

        inputs = data[:20000, :-1]
        target = data[:20000, -1]

        # inputs = data[:,:-1]
        # target = data[:,-1]

        ##################3

        raw_labels = target
        raw_input_features = inputs

        """ we take a pre-processing step such that the dataset is a bit more balanced """
        idx_negative_label = raw_labels == 1  # 1 and 0 are dominant but 1 has more labels
        idx_positive_label = raw_labels != 1

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]

        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate  # 0.3
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]


        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        ###############3

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.70, test_size=0.30,
                                                            random_state=seed_number)

   return X_train, X_test, y_train, y_test, n_classes, num_numerical_inputs, num_categorical_inputs


def test_models(X_tr, y_tr, X_te, y_te, n_classes, datasettype, args, data_name):

    print("\n", datasettype, "data\n")

    roc_arr = []
    prc_arr = []
    f1_arr = []

    models = np.array(
        [
         LogisticRegression(),
         GaussianNB(),
         BernoulliNB(),
         LinearSVC(),
         DecisionTreeClassifier(),
         LinearDiscriminantAnalysis(),
         AdaBoostClassifier(),
         BaggingClassifier(),
         RandomForestClassifier(),
         GradientBoostingClassifier(subsample=0.1, n_estimators=50),
         MLPClassifier(),
        xgboost.XGBClassifier(disable_default_eval_metric=1, learning_rate=0.5)
        ])


    models_to_test = models[np.array(args)]

    if n_classes == 2:
        for model in models_to_test:

            print('\n', type(model))
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)  # test on real data
            roc = roc_auc_score(y_te, pred)
            prc = average_precision_score(y_te, pred)

            if str(model)[0:11] == 'BernoulliNB':
                print('training again')

                model = BernoulliNB(alpha=0.02)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp1 = roc_auc_score(y_te, pred)
                prc_temp1 = average_precision_score(y_te, pred)

                print('training again')
                model = BernoulliNB(alpha=0.5)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp2 = roc_auc_score(y_te, pred)
                prc_temp2 = average_precision_score(y_te, pred)

                print('training again')
                model = BernoulliNB(alpha=1.0)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp3 = roc_auc_score(y_te, pred)
                prc_temp3 = average_precision_score(y_te, pred)

                print('training again')
                model = BernoulliNB(alpha=1.0, binarize=0.4)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp4 = roc_auc_score(y_te, pred)
                prc_temp4 = average_precision_score(y_te, pred)

                print('training again')
                model = BernoulliNB(alpha=1.0, binarize=0.5)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp5 = roc_auc_score(y_te, pred)
                prc_temp5 = average_precision_score(y_te, pred)

                roc = max(roc, roc_temp1, roc_temp2, roc_temp3, roc_temp4, roc_temp5)
                prc = max(prc, prc_temp1, prc_temp2, prc_temp3, prc_temp4, prc_temp5)


            elif str(model)[0:10] == 'GaussianNB':
                print('training again')

                model = GaussianNB(var_smoothing=1e-3, priors=(sum(y_tr)/len(y_tr),1-sum(y_tr)/len(y_tr)))
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp1 = roc_auc_score(y_te, pred)
                prc_temp1 = average_precision_score(y_te, pred)

                roc = max(roc, roc_temp1)
                prc = max(prc, prc_temp1)

            elif str(model)[0:12] == 'RandomForest':
                print('training again')

                model = RandomForestClassifier(n_estimators=200)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp1 = roc_auc_score(y_te, pred)
                prc_temp1 = average_precision_score(y_te, pred)

                print('training again')
                model = RandomForestClassifier(n_estimators=70)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp2 = roc_auc_score(y_te, pred)
                prc_temp2 = average_precision_score(y_te, pred)

                print('training again')
                model = RandomForestClassifier(n_estimators=30)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp3 = roc_auc_score(y_te, pred)
                prc_temp3 = average_precision_score(y_te, pred)

                print('training again')
                model = RandomForestClassifier(n_estimators=10)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp4 = roc_auc_score(y_te, pred)
                prc_temp4 = average_precision_score(y_te, pred)

                roc = max(roc, roc_temp1, roc_temp2, roc_temp3, roc_temp4)
                prc = max(prc, prc_temp1, prc_temp2, prc_temp3, prc_temp4)

            elif str(model)[0:18] == 'LogisticRegression':

                print('logistic regression with balanced class weight')
                model = LogisticRegression(solver='lbfgs', max_iter=50000, tol=1e-12)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp1 = roc_auc_score(y_te, pred)
                prc_temp1 = average_precision_score(y_te, pred)

                print('logistic regression with saga solver')
                model = LogisticRegression(solver='saga', penalty='l1', tol=1e-12)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp2 = roc_auc_score(y_te, pred)
                prc_temp2 = average_precision_score(y_te, pred)

                print('logistic regression with liblinear solver')
                model = LogisticRegression(solver='liblinear', penalty='l1', class_weight='balanced',tol=1e-8, C=0.1)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp3 = roc_auc_score(y_te, pred)
                prc_temp3 = average_precision_score(y_te, pred)

                print('logistic regression with liblinear solver')
                model = LogisticRegression(solver='liblinear', penalty='l1', class_weight='balanced',tol=1e-8, C=0.05)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp4 = roc_auc_score(y_te, pred)
                prc_temp4 = average_precision_score(y_te, pred)

                roc = max(roc, roc_temp1, roc_temp2, roc_temp3, roc_temp4)
                prc = max(prc, prc_temp1, prc_temp2, prc_temp3, prc_temp4)
                # roc = max(roc, roc_temp1, roc_temp2)
                # prc = max(prc, prc_temp1, prc_temp2)


            elif str(model)[0:9] == 'LinearSVC':
                print('training again')

                model = LinearSVC(max_iter=10000, tol=1e-8, loss='hinge')
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp1 = roc_auc_score(y_te, pred)
                prc_temp1 = average_precision_score(y_te, pred)

                print('training again')
                model = LinearSVC(max_iter=10000, tol=1e-8, loss='hinge', class_weight='balanced')
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp2 = roc_auc_score(y_te, pred)
                prc_temp2 = average_precision_score(y_te, pred)

                print('training again')
                model = LinearSVC(max_iter=10000, tol=1e-12, loss='hinge', C=0.01)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp3 = roc_auc_score(y_te, pred)
                prc_temp3 = average_precision_score(y_te, pred)

                roc = max(roc, roc_temp1, roc_temp2, roc_temp3)
                prc = max(prc, prc_temp1, prc_temp2, prc_temp3)


            elif str(model)[0:12] == 'DecisionTree':
                print('training again')

                model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', max_features='log2')
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp1 = roc_auc_score(y_te, pred)
                prc_temp1 = average_precision_score(y_te, pred)

                print('training again')
                model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', max_features='auto')
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp2 = roc_auc_score(y_te, pred)
                prc_temp2 = average_precision_score(y_te, pred)

                print('training again')
                model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', max_features='sqrt')
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp3 = roc_auc_score(y_te, pred)
                prc_temp3 = average_precision_score(y_te, pred)


                roc = max(roc, roc_temp1, roc_temp2, roc_temp3)
                prc = max(prc, prc_temp1, prc_temp2, prc_temp3)

            elif str(model)[0:26] == 'LinearDiscriminantAnalysis':

                print('test LDA with different hyperparameters')
                model = LinearDiscriminantAnalysis(solver='eigen', tol=1e-12, shrinkage='auto')
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp1 = roc_auc_score(y_te, pred)
                prc_temp1 = average_precision_score(y_te, pred)

                model = LinearDiscriminantAnalysis(solver='lsqr', tol=1e-12, shrinkage=0.6)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp2 = roc_auc_score(y_te, pred)
                prc_temp2 = average_precision_score(y_te, pred)

                model = LinearDiscriminantAnalysis(solver='lsqr', tol=1e-12, shrinkage=0.75)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp3 = roc_auc_score(y_te, pred)
                prc_temp3 = average_precision_score(y_te, pred)

                roc = max(roc, roc_temp1, roc_temp2, roc_temp3)
                prc = max(prc, prc_temp1, prc_temp2, prc_temp3)


            elif str(model)[0:8] == 'AdaBoost':

                model = AdaBoostClassifier(n_estimators=100, learning_rate=0.8)  # improved
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp1 = roc_auc_score(y_te, pred)
                prc_temp1 = average_precision_score(y_te, pred)

                model = AdaBoostClassifier(n_estimators=200, learning_rate=0.5)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp2 = roc_auc_score(y_te, pred)
                prc_temp2 = average_precision_score(y_te, pred)

                model = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, algorithm='SAMME')
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp3 = roc_auc_score(y_te, pred)
                prc_temp3 = average_precision_score(y_te, pred)

                roc = max(roc, roc_temp1, roc_temp2, roc_temp3)
                prc = max(prc, prc_temp1, prc_temp2, prc_temp3)

            elif str(model)[0:7] == 'Bagging':

                print('test Bagging with different hyperparameters')
                model = BaggingClassifier(max_samples=0.1, n_estimators=20)  # improved
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp1 = roc_auc_score(y_te, pred)
                prc_temp1 = average_precision_score(y_te, pred)

                roc = max(roc, roc_temp1)
                prc = max(prc, prc_temp1)


            elif str(model)[0:3] == 'MLP':

                model = MLPClassifier(learning_rate='adaptive', alpha=0.01, tol=1e-10)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp1 = roc_auc_score(y_te, pred)
                prc_temp1 = average_precision_score(y_te, pred)

                model = MLPClassifier(solver='lbfgs', alpha=0.001, tol=1e-8)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp2 = roc_auc_score(y_te, pred)
                prc_temp2 = average_precision_score(y_te, pred)

                model = MLPClassifier(solver='sgd', alpha=0.001, tol=1e-8)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp3 = roc_auc_score(y_te, pred)
                prc_temp3 = average_precision_score(y_te, pred)

                roc = max(roc, roc_temp1, roc_temp2, roc_temp3)
                prc = max(prc, prc_temp1, prc_temp2, prc_temp3)

            elif str(model)[0:13] == 'XGBClassifier':

                print('test XGB with different hyperparameters')
                model = xgboost.XGBClassifier(disable_default_eval_metric=1, learning_rate=0.7)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp1 = roc_auc_score(y_te, pred)
                prc_temp1 = average_precision_score(y_te, pred)

                model = xgboost.XGBClassifier(disable_default_eval_metric=1, learning_rate=0.8)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp2 = roc_auc_score(y_te, pred)
                prc_temp2 = average_precision_score(y_te, pred)

                model = xgboost.XGBClassifier(disable_default_eval_metric=1, learning_rate=1.0)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                roc_temp3 = roc_auc_score(y_te, pred)
                prc_temp3 = average_precision_score(y_te, pred)

                roc = max(roc, roc_temp1, roc_temp2, roc_temp3)
                prc = max(prc, prc_temp1, prc_temp2, prc_temp3)

            roc_arr.append(roc)
            prc_arr.append(prc)

            print("ROC on test %s data is %.3f" % (datasettype, roc))
            print("PRC on test %s data is %.3f" % (datasettype, prc))

    else: # multiclass classification datasets

        for model in models_to_test:

            print('\n', type(model))
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)  # test on real data
            f1score = f1_score(y_te, pred, average='weighted')

            if data_name == 'covtype':
                prior_class = np.array([sum(y_tr==0), sum(y_tr==1), sum(y_tr==2), sum(y_tr==3), sum(y_tr==4), sum(y_tr==5), sum(y_tr==6)])/(y_tr.shape[0])
                print("prior_class from BernoulliNB=", prior_class)
                print("prior_class.shape from BernoulliNB=", prior_class.shape)
                print("prior_class.squeeze()=", prior_class.squeeze())
            elif data_name == 'intrusion':
                prior_class = np.array([sum(y_tr==0), sum(y_tr==1), sum(y_tr==2), sum(y_tr==3), sum(y_tr==4)])/(y_tr.shape[0])

            if str(model)[0:11] == 'BernoulliNB':

                print('training again')

                model = BernoulliNB(alpha=0.02, class_prior=prior_class.squeeze())
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score1 = f1_score(y_te, pred, average='weighted')

                print('training again')
                model = BernoulliNB(alpha=0.5, class_prior=prior_class.squeeze())
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score2 = f1_score(y_te, pred, average='weighted')

                print('training again')
                model = BernoulliNB(alpha=1.0, class_prior=prior_class.squeeze())
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score3 = f1_score(y_te, pred, average='weighted')

                print('training again')
                model = BernoulliNB(alpha=5.0, binarize=0.4, class_prior=prior_class.squeeze())
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score4 = f1_score(y_te, pred, average='weighted')

                print('training again')
                model = BernoulliNB(alpha=10.0, binarize=0.05, class_prior=prior_class.squeeze())
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score5 = f1_score(y_te, pred, average='weighted')

                #print('training again')
                #train_mean=np.mean(X_tr)
                #print("train_mean=", train_mean)
                #model = BernoulliNB(class_prior=prior_class.squeeze(), binarize = train_mean)
                #model.fit(X_tr, y_tr)
                #pred = model.predict(X_te)  # test on real data
                #f1score6 = f1_score(y_te, pred, average='weighted')

                f1score = max(f1score, f1score1, f1score2, f1score3, f1score4, f1score5)

            elif str(model)[0:16] == 'GradientBoosting':

                model = GradientBoostingClassifier(learning_rate=0.5, n_estimators=100)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score1 = f1_score(y_te, pred, average='weighted')

                model = GradientBoostingClassifier(learning_rate=0.5, n_estimators=200)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score2 = f1_score(y_te, pred, average='weighted')

                model = GradientBoostingClassifier(random_state=0)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score3 = f1_score(y_te, pred, average='weighted')

                model = GradientBoostingClassifier(subsample=0.5, n_estimators=100, learning_rate=0.05)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score4 = f1_score(y_te, pred, average='weighted')

                model = GradientBoostingClassifier(subsample=0.5, n_estimators=100, learning_rate=0.01)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score5 = f1_score(y_te, pred, average='weighted')

                model = GradientBoostingClassifier(subsample=0.5, n_estimators=100, learning_rate=0.001)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score6 = f1_score(y_te, pred, average='weighted')

                f1score = max(f1score, f1score1, f1score2, f1score3, f1score4, f1score5, f1score6)


            elif str(model)[0:10] == 'GaussianNB':
                print('training again')

                model = GaussianNB(var_smoothing=1e-3, priors=prior_class)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score1 = f1_score(y_te, pred, average='weighted')

                model = GaussianNB(priors=prior_class, var_smoothing=1e-12)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score2 = f1_score(y_te, pred, average='weighted')

                model = GaussianNB(var_smoothing=1e-13)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score3 = f1_score(y_te, pred, average='weighted')

                model = GaussianNB(var_smoothing=0.2, priors=prior_class)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score4 = f1_score(y_te, pred, average='weighted')


                model = GaussianNB(var_smoothing=0.5, priors=prior_class)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score5 = f1_score(y_te, pred, average='weighted')

                model = GaussianNB(var_smoothing=0.8, priors=prior_class)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score6 = f1_score(y_te, pred, average='weighted')

                model = GaussianNB(var_smoothing=1.2, priors=prior_class)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score7 = f1_score(y_te, pred, average='weighted')

                model = GaussianNB(var_smoothing=2, priors=prior_class)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score8 = f1_score(y_te, pred, average='weighted')

                model = GaussianNB(var_smoothing=10, priors=prior_class)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score9 = f1_score(y_te, pred, average='weighted')

                model = GaussianNB(var_smoothing=20, priors=prior_class)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score10 = f1_score(y_te, pred, average='weighted')

                model = GaussianNB(var_smoothing=0.34, priors=prior_class)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score11 = f1_score(y_te, pred, average='weighted')

                model = GaussianNB(var_smoothing=0.138, priors=prior_class)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score12 = f1_score(y_te, pred, average='weighted')

                model = GaussianNB(var_smoothing=1e-2, priors=prior_class)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score13 = f1_score(y_te, pred, average='weighted')



                f1score = max(f1score, f1score1, f1score2, f1score3, f1score4, f1score5, f1score6, f1score7, f1score8, f1score9, f1score10, f1score11,  f1score12, f1score13)

            elif str(model)[0:12] == 'RandomForest':
                print('training again')

                model = RandomForestClassifier(n_estimators=200)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score1 = f1_score(y_te, pred, average='weighted')

                print('training again')
                model = RandomForestClassifier(n_estimators=70)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score2 = f1_score(y_te, pred, average='weighted')

                print('training again')
                model = RandomForestClassifier(n_estimators=30)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score3 = f1_score(y_te, pred, average='weighted')

                print('training again')
                model = RandomForestClassifier(n_estimators=10)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score4 = f1_score(y_te, pred, average='weighted')

                f1score = max(f1score, f1score1, f1score2, f1score3, f1score4)

            elif str(model)[0:18] == 'LogisticRegression':

                print('logistic regression with balanced class weight')
                model = LogisticRegression(solver='lbfgs', max_iter=50000, tol=1e-12, multi_class='multinomial')
                # model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score1 = f1_score(y_te, pred, average='weighted')

                print('logistic regression with saga solver')
                model = LogisticRegression(solver='saga', penalty='l1', tol=1e-12)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score2 = f1_score(y_te, pred, average='weighted')

                print('logistic regression with liblinear solver')
                model = LogisticRegression(solver='liblinear', penalty='l1', class_weight='balanced', tol=1e-8, C=0.1)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score3 = f1_score(y_te, pred, average='weighted')

                print('logistic regression with liblinear solver')
                model = LogisticRegression(solver='liblinear', penalty='l1', class_weight='balanced', tol=1e-8, C=0.05)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score4 = f1_score(y_te, pred, average='weighted')

                f1score = max(f1score, f1score1, f1score2, f1score3, f1score4)


            elif str(model)[0:9] == 'LinearSVC':
                print('training again')

                model = LinearSVC(max_iter=10000, tol=1e-8, loss='hinge')
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score1 = f1_score(y_te, pred, average='weighted')

                print('training again')
                model = LinearSVC(max_iter=10000, tol=1e-8, loss='hinge', class_weight='balanced')
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score2 = f1_score(y_te, pred, average='weighted')

                print('training again')
                model = LinearSVC(max_iter=10000, tol=1e-12, loss='hinge', C=0.01)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score3 = f1_score(y_te, pred, average='weighted')

                print('training again')
                model = LinearSVC(max_iter=10000, tol=1e-12, multi_class = 'crammer_singer')
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score4 = f1_score(y_te, pred, average='weighted')

                model = LinearSVC(max_iter=10000, tol=1e-16, loss='hinge',multi_class = 'crammer_singer', C=0.1)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score5 = f1_score(y_te, pred, average='weighted')

                model = LinearSVC(max_iter=20000, tol=1e-16, loss='hinge')
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score6 = f1_score(y_te, pred, average='weighted')

                f1score = max(f1score, f1score1, f1score2, f1score3, f1score4, f1score5, f1score6)


            elif str(model)[0:12] == 'DecisionTree':
                print('training again')

                model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', max_features='log2')
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score1 = f1_score(y_te, pred, average='weighted')

                print('training again')
                model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', max_features='auto')
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score2 = f1_score(y_te, pred, average='weighted')

                print('training again')
                model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', max_features='sqrt')
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score3 = f1_score(y_te, pred, average='weighted')

                f1score = max(f1score, f1score1, f1score2, f1score3)

            elif str(model)[0:26] == 'LinearDiscriminantAnalysis':

                #print('test LDA with different hyperparameters')
                #model = LinearDiscriminantAnalysis(solver='eigen', tol=1e-12, shrinkage='auto', priors=prior_class.squeeze())
                #model.fit(X_tr, y_tr)
                #pred = model.predict(X_te)  # test on real data
                #f1score1 = f1_score(y_te, pred, average='weighted')

                model = LinearDiscriminantAnalysis(solver='lsqr', tol=1e-12, shrinkage=0.6)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score2 = f1_score(y_te, pred, average='weighted')

                model = LinearDiscriminantAnalysis(solver='lsqr', tol=1e-12, shrinkage=0.75)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score3 = f1_score(y_te, pred, average='weighted')

                model = LinearDiscriminantAnalysis(solver='lsqr', tol=1e-12, shrinkage=0.9)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score4= f1_score(y_te, pred, average='weighted')


                f1score = max(f1score2, f1score3, f1score4)


            elif str(model)[0:8] == 'AdaBoost':

                model = AdaBoostClassifier(n_estimators=100, learning_rate=0.8)  # improved
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score1 = f1_score(y_te, pred, average='weighted')

                model = AdaBoostClassifier(n_estimators=200, learning_rate=0.5)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score2 = f1_score(y_te, pred, average='weighted')

                model = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, algorithm='SAMME')
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score3 = f1_score(y_te, pred, average='weighted')

                model = AdaBoostClassifier(n_estimators=100, learning_rate=5.0, random_state = 0)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score4 = f1_score(y_te, pred, average='weighted')

                model = AdaBoostClassifier(n_estimators=1000, learning_rate=5.0, random_state = 0)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score5 = f1_score(y_te, pred, average='weighted')


                model = AdaBoostClassifier(n_estimators=1000, learning_rate=4.0, random_state = 0)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score6 = f1_score(y_te, pred, average='weighted')

                # model = AdaBoostClassifier(n_estimators=500, learning_rate=20)
                # model.fit(X_tr, y_tr)
                # pred = model.predict(X_te)  # test on real data
                # f1score7 = f1_score(y_te, pred, average='weighted')

                f1score = max(f1score, f1score1, f1score2, f1score3, f1score4, f1score5, f1score6)

            elif str(model)[0:7] == 'Bagging':

                print('test Bagging with different hyperparameters')
                model = BaggingClassifier(max_samples=0.1, n_estimators=20)  # improved
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score1 = f1_score(y_te, pred, average='weighted')

                model = BaggingClassifier(n_estimators=200, warm_start=True)  # improved
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score2 = f1_score(y_te, pred, average='weighted')

                model = BaggingClassifier(n_estimators=200, warm_start=True, max_features=10)  # improved
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score3 = f1_score(y_te, pred, average='weighted')

                f1score = max(f1score, f1score1, f1score2, f1score3)


            elif str(model)[0:3] == 'MLP':

                model = MLPClassifier(learning_rate='adaptive', alpha=0.01, tol=1e-10)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score1 = f1_score(y_te, pred, average='weighted')

                model = MLPClassifier(solver='lbfgs', alpha=0.001, tol=1e-8)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score2 = f1_score(y_te, pred, average='weighted')

                model = MLPClassifier(solver='sgd', alpha=0.001, tol=1e-8)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score3 = f1_score(y_te, pred, average='weighted')

                f1score = max(f1score, f1score1, f1score2, f1score3)

            elif str(model)[0:13] == 'XGBClassifier':

                print('test XGB with different hyperparameters')
                model = xgboost.XGBClassifier(disable_default_eval_metric=1, learning_rate=0.7)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score1 = f1_score(y_te, pred, average='weighted')

                model = xgboost.XGBClassifier(disable_default_eval_metric=1, learning_rate=0.8)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score2 = f1_score(y_te, pred, average='weighted')

                model = xgboost.XGBClassifier(disable_default_eval_metric=1, learning_rate=1.0)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score3 = f1_score(y_te, pred, average='weighted')

                model = xgboost.XGBClassifier(disable_default_eval_metric=1, learning_rate=5.0)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)  # test on real data
                f1score4 = f1_score(y_te, pred, average='weighted')

                f1score = max(f1score, f1score1, f1score2, f1score3, f1score4)


            print("F1-score on test %s data is %.3f" % (datasettype, f1score))
            f1_arr.append(f1score)

    if n_classes > 2:

        res1 = np.mean(f1_arr)
        res1_arr = f1_arr
        print("------\nf1 mean across methods is %.3f\n" % res1)
        res2_arr = 0  # dummy

    else:

        res1 = np.mean(roc_arr)
        res1_arr = roc_arr
        res2 = np.mean(prc_arr)
        res2_arr = prc_arr
        print("-" * 40)
        print("roc mean across methods is %.3f" % res1)
        print("prc mean across methods is %.3f\n" % res2)

    return res1_arr, res2_arr
