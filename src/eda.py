# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 11:50:05 2019
Some of the variables and docstrings are anonymized to protect the company's business
@author: dianashao
"""
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from enum import Enum
import json
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

#pd.set_option('display.float_format', lambda x: '%.3f' % x)

from data_cleaning import *

class_names = ['negative', 'positive']

class Stage(Enum):
    DataCleaning = 1
    Training = 2
    Prediction = 3

def feature_engineer(master):
    '''
    Generate features that might explain the outcome.
    1. Days between "Submit" and "PriorSubmit" to control for recent activity
    or potential liquidity constraint.
    2. Waitlist with some specific values means they were not "waitlisted", construct a
    dummy variable based on this variable.
    3. Identify the physical location of the properties.
    4. Fill in dollar variables with 0 to indicate there was 0 activity from an user
    to that given property.
    '''
    df = master.copy()
    df['managerId'] = df['managerId'].astype('category')
    df['new_region'] = df['new_region'].astype('category')
    df['deal_region'] = df['deal_region'].astype('category')
    df['zipcode'] = df['zipcode'].astype('category')
    df['userrelation'] = df['userrelation'].astype('category')

    return df

def plot_roc(y_test, y_pred, name="roc_plot"):
    #Print Area Under Curve
    plt.figure()
    false_positive_rate, recall, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, recall)
    plt.rc('font', size = 18)
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()
    plt.savefig(name)

def plot_train_eval_history(training_metric, validation_metric, metric_name = "AUC", 
                            filename="train_valid_history"):
    plt.figure()
    plt.rc('font', size = 18)
    plt.title('Training/Validation %s History' %metric_name)
    plt.plot(training_metric, 'b', label = 'Training %s' %metric_name)
    plt.plot(validation_metric, 'r', label = 'Validation %s' %metric_name)
    plt.legend(loc='lower right')
    #plt.ylim([0.0,1.0])
    plt.ylabel(metric_name, fontsize=18)
    plt.xlabel('Epoch')
    #plt.show()
    plt.savefig(filename)
    
def plot_confusion_matrix(y_true, y_pred_binary, classes, normalize=False, title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    # Only use the labels that appear in the data
    classes = ['negative','positive']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes)
    plt.rc('font', size = 20)
    ax.set_title(title)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="center")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def get_seleted_features():
    features = ['list of variables selected for the model']
    
    category_features = ['list of categorical variables selected for the model']
    
    return features, category_features

def model_pipeline(master_feature, run_grid_search = False, experiment_name = ''):
    '''
    Preprocess the data, and train the model.
    1. The outcome will be fairly unbalanced because most of the time only a small portion
        of the potential users act in a deal.
    2. Use lightGBM to handle the imbalanced data, specify categorical variables as
        categorical variables, don't include userID and propertyID.
    '''
    df = master_feature.copy()
    features, category = get_seleted_features()
    X = df[features]    
    y = df['acted'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 27)

    lgb_train = lgb.Dataset(X_train, categorical_feature=category,label=y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {}
    params['learning_rate'] = 0.003
    params['boosting_type'] = 'gbdt'
    params['bagging_freq'] = 1
    params['bagging_fraction'] = 0.7
    params['objective'] = 'binary'
    params['metric'] = 'auc'
    params['is_unbalance'] = False
    params['n_estimators'] = 1000
    
    if run_grid_search:
        grid_params = {}
        grid_params['num_leaves'] = [10, 20, 40, 80, 100, 200, 500]
        grid_params['max_depth'] = [2, 4, 8, 16, 20, 30, 40, 50]
        grid_params['scale_pos_weight'] = [1, 10, 40, 80, 100, 200, 400, 800, 1000]
        grid_params['sub_feature'] = [0.4, 0.5, 0.6, 0.7, 0.8]
        grid_params['subsample'] = [0.4, 0.5, 0.6, 0.7, 0.8]
    
        classifier = lgb.LGBMClassifier(**params)
    
        # View the default model params:
        classifier.get_params().keys()
    
        # Create the grid
        grid = GridSearchCV(classifier, grid_params, verbose=1, cv=4, n_jobs=-1)
        grid.fit(X_train, y_train)
    
        print(grid.best_params_)
        print(grid.best_score_)
    
        # Final training
        params.update(grid.best_params_)
    else:
        params['num_leaves'] = 80
        params['max_depth'] = 16
        params['scale_pos_weight'] = 50
        params['sub_feature'] = 0.4
        params['subsample'] = 0.8
        params['early_stopping_rounds'] = 30
        
    evals_result = {}
    clf = lgb.train(params, lgb_train, valid_sets=[lgb_train,lgb_eval], valid_names = ['train', 'valid'], 
                    verbose_eval=10, evals_result=evals_result)

    # Dump the training parameter
    print(params)
    with open(experiment_name+'training_params.json', 'w') as fp:
        json.dump(params, fp)

    training_history = evals_result['train']['auc']
    validation_history = evals_result['valid']['auc']

    plot_train_eval_history(training_history, validation_history, 
                            filename=experiment_name+'train_valid_history')

    #Prediction
    y_pred=clf.predict(X_test)

    plot_roc(y_test, y_pred, name=experiment_name+'roc_plot')

    #convert into binary values
    y_pred_bi = np.where(y_pred>0.5, 1, 0)

    #Confusion matrix
    cm = confusion_matrix(y_test, y_pred_bi)
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred_bi, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred_bi, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.show()
    
    #Accuracy
    accuracy=accuracy_score(y_pred_bi,y_test)
    print(str(np.round(accuracy, 4)) + " is the accuracy rate.")
    print('Saving model...')
    clf.save_model(experiment_name+'.txt')

    return clf, cm, accuracy, X_test, y_test, y_pred, y_pred_bi

def baseline_model(master_feature, experiment_name='baseline'):
    '''
    Run a baseline logistic regression model.
    1. The outcome will be fairly unbalanced because most of the time only a small portion
        of the potential users act in a deal.
    '''
    df = master_feature.copy()
    y = df['acted'].values
    category_feature = ['list of categorical variables without feature engineering']
    useful_feature = ['list of variables without feature engineering'] 
    df = df[useful_feature]
    for var in category_feature:
        cat_list ='var'+'_'+var
        cat_list = pd.get_dummies(df[var], prefix=var)
        df1 = df.join(cat_list)
        df = df1
        
    df_vars=df.columns.values.tolist()
    to_keep=[i for i in df_vars if i not in category_feature]
    
    X = df[to_keep].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)

#   evals_result = {}
    baseline_model = LogisticRegression(class_weight='balanced')
    baseline_model.fit(X_train, y_train)

#   training_history = evals_result['train']['auc']
#   validation_history = evals_result['valid']['auc']
#
#   plot_train_eval_history(training_history, validation_history, 
#                            filename=experiment_name+'train_valid_history')
    #Prediction
    y_pred=baseline_model.predict_proba(X_test)[:,1]

    plot_roc(y_test, y_pred, name=experiment_name+'roc_plot')

    #convert into binary values
    y_pred_binary = np.where(y_pred>0.5, 1, 0)

    #Confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    #Accuracy
    accuracy=accuracy_score(y_pred_binary,y_test)
    print(str(np.round(accuracy, 4)) + " is the accuracy rate.")

    return baseline_model, cm, accuracy, X_test, y_test, y_pred

def live_demo(master_feature, clf, name):
    deal = str(name).lower()
    X_userID = master_feature[master_feature['propertyId'] == deal]['UserId'].reset_index(drop=True)
    actual_act = master_feature[master_feature['propertyId']==deal]['acted'].reset_index(drop=True)
    X_input = master_feature[master_feature['propertyId'] == deal].reset_index(drop=True)
    features, category = get_seleted_features()
    X_input = X_input[features]
    #Prediction
    y_pred=np.round(clf.predict(X_input),2)
    #convert into binary values, might need to change threshold
    y_binary = np.where(y_pred>0.4, 1, 0)

    prediction = pd.DataFrame(columns=['UserId','act_Prediction','act_Probability','actual_act'])
    prediction['UserId'] = X_userID
    prediction['act_Prediction'] = pd.Series(y_binary)
    prediction['act_Probability'] = pd.Series(y_pred)
    prediction['actual_act'] = actual_act
    prediction = prediction.drop_duplicates()
    # print("There are " + str(prediction['act_Prediction'].sum()) + " potential users", end=' ')
    # print("for this deal.")
    # print("======================================================================")
    # print("Start contacting the following Users: ")
    # print('========================================================================')
    # print([str(ele) for ele in prediction[prediction['act_Prediction']==1]['UserId']])
    return prediction.sort_values(by=['actual_act','act_Probability'],ascending=[False,False])


if __name__ == '__main__':
    master_location = r"uploads\master.csv"
    stage = Stage.Training
    
    if stage == Stage.Training:
        print("Start loading master file")
        master = pd.read_csv(master_location, index_col=0)
        
        print("Start engineering features")
        master_feature = feature_engineer(master)

        print("Start training")
        lgb_model, matrix, accuracy, x_test, y_test, y_pred, y_pred_bi = model_pipeline(
                master_feature,
                experiment_name='training_visualization\\manager_userZip_dealview')
    elif stage == Stage.Prediction:
        print("Start loading master file")
        master = pd.read_csv(master_location, index_col=0)
        
        print("Start engineering features")
        master_feature = feature_engineer(master)
        
        print("Start loading model")
        clf = lgb.Booster(model_file=r"models\manager_userZip_dealview.txt")
        
        name = '5d31dde054734746bc600c2d7c92a4e7'
        actment_prediction = live_demo(master_feature, clf, name)
          
        
import shap
shap_values = shap.TreeExplainer(lgb_model).shap_values(x_test)
feature_importance = shap.summary_plot(shap_values, x_test)       
        
        
        
        
        
        
        
        
        