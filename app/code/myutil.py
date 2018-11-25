import pandas as pd
import numpy as np
import math
import random
import time
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import lightgbm as lgb
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
warnings.filterwarnings('ignore')

def get_train_test_label(row=None):
    train = pd.read_csv("main_table.csv", nrows = row)
    test = pd.read_csv("test_table.csv", nrows = row)
    label = train['TARGET']
    print('train datasset shape:',train.shape)
    print('test datasset shape:',test.shape)
    return train, test, label

# undersample without replacement, return a list of undersampled df
def undersample(df):
    pos = df[df.TARGET == 1]
    neg = df[df.TARGET == 0]
    shuffled_neg = shuffle(neg)

    neg_list = []
    for i in range(len(neg) // len(pos)):
        neg_list.append(shuffled_neg[i*len(pos):(i+1)*len(pos)])

    undersampled_list = []
    for neg_df in neg_list:
        undersampled_df = neg_df.append(pos, ignore_index=True)
        undersampled_list.append(shuffle(undersampled_df).reset_index(drop=True))
    return undersampled_list

def feature_select_cross_validation_undersample(feature_importance_df_, data_, test_, y_, model):
    cols_all = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False).index
    
    itr_num = int(math.floor(len(cols_all)/50))
    print("number of iteration:", itr_num)
    
    for i in range(itr_num):
        cols = cols_all[:(i*50)]
        data_sf = data_[cols]
        data_sf[['TARGET','Unnamed: 0']] = data_[['TARGET','Unnamed: 0']]
#         print(i," selected index:",cols)
        print(i)
        oof_preds, test_preds, importances, folds = cross_validation_undersample(data_sf, test_, y_, model)
        test_preds.to_csv('{}_{}_submission_undersample.csv'.format(str(model),i), index=False)

def cross_validation_undersample(data_, test_, y_, model):
    
    
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    
    udf = undersample(data_)
    folds_ = KFold(n_splits=len(udf), shuffle=True, random_state=546789)
    feature_importance_df = pd.DataFrame()
    
    feats = [f for f in data_.columns if f not in ['SK_ID_CURR','TARGET','Unnamed: 0']]
    print('total n fold :',len(udf))
    for n_fold, u in enumerate(udf):
        
        idx = np.arange(len(u.index))
        np.random.shuffle(idx)
        trn_idx = idx[0:math.floor(len(idx)*0.75)]
        val_idx = idx[math.floor(len(idx)*0.75):len(idx)]
        #print(trn_idx)
        #print('u shape:', u.shape)
        label = u['TARGET']         
        trn_x = u[feats].iloc[trn_idx]
        trn_y = label.iloc[trn_idx]
        val_x = u[feats].iloc[val_idx]
        val_y = label.iloc[val_idx]
        #print(trn_x)
        #print('train_x shape:', trn_x)
        #print('val_x shape:', val_x.shape)
           
         # Make the new model 
        model.init_model()
        
        # Train on the training data
        model.train(trn_x, trn_y)

        # Make predictions
        # Make sure to select the second column only    
        #print('oof_preds has shape:', oof_preds[val_idx].shape)
        #'Unnamed: 0' is same as the idx in the original data
        true_val_idx = u['Unnamed: 0'].iloc[val_idx].values.flat
        oof_preds[true_val_idx] = model.predict(val_x)
        sub_preds +=  model.predict(test_[feats])/ len(udf)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        
        #print(fold_importance_df.shape, log_reg.coef_.shape)
        
        fold_importance_df["importance"] = model.get_coef()
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        #print(val_y.shape, oof_preds[val_idx].shape)
        #print(val_y, oof_preds[val_idx])
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del trn_x, trn_y, val_x, val_y
        gc.collect()
        
    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds)) 
    
    test_['TARGET'] = sub_preds
    #test_[['SK_ID_CURR', 'TARGET']].to_csv('{}_submission_undersample.csv'.format(model), index=False, float_format='%.8f')
    return oof_preds, test_[['SK_ID_CURR', 'TARGET']], feature_importance_df, folds_

def cross_validation(data_, test_, y_, model):
    folds_ = KFold(n_splits=5, shuffle=True, random_state=546789)
    
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    
    feature_importance_df = pd.DataFrame()
    
    feats = [f for f in data_.columns if f not in ['SK_ID_CURR','TARGET','Unnamed: 0']]
    
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        
         # Make the new model 
        model.init_model()
        
        # Train on the training data
        model.train(trn_x, trn_y)

        # Make predictions
        # Make sure to select the second column only    
        #print('oof_preds has shape:', oof_preds[val_idx].shape)
        oof_preds[val_idx] = model.predict(val_x)
        sub_preds +=  model.predict(test_[feats])/ folds_.n_splits
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        
        #print(fold_importance_df.shape, log_reg.coef_.shape)
        
        fold_importance_df["importance"] = model.get_coef()
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        #print(val_y.shape, oof_preds[val_idx].shape)
        #print(val_y, oof_preds[val_idx])
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del trn_x, trn_y, val_x, val_y
        gc.collect()
        
    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds)) 
    
    test_['TARGET'] = sub_preds
    #test_[['SK_ID_CURR', 'TARGET']].to_csv('{}_submission_undersample.csv'.format(model), index=False, float_format='%.8f')
    return oof_preds, test_[['SK_ID_CURR', 'TARGET']], feature_importance_df, folds_

def display_importances(feature_importance_df_,title):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].reset_index()
    plt.figure(figsize=(8,10))
    sns.barplot(x="importance", y="feature", 
                data=cols)
    plt.title('{} Features '.format(title))
    plt.tight_layout()
    plt.savefig('{}_importances.png'.format(title))

def display_roc_curve(y_, oof_preds_, folds_idx_,title, is_with_nfold):
    # Plot ROC curves
    plt.figure(figsize=(6,6))
    scores = [] 
    if(is_with_nfold):
        for n_fold, (_, val_idx) in enumerate(folds_idx_):  
            # Plot the roc curve
            fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
            score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])
            scores.append(score)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)
    score = roc_auc_score(y_, oof_preds_)
    plt.plot(fpr, tpr, color='b',
             label='Full ROC (AUC = %0.4f)' % (score),
             lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC Curve'.format(title))
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    plt.savefig('{}_roc_curve.png'.format(title))
    
def display_precision_recall(y_, oof_preds_, folds_idx_, title, is_with_nfold):
    # Plot ROC curves
    plt.figure(figsize=(6,6))
    
    scores = [] 
    if is_with_nfold:
        for n_fold, (_, val_idx) in enumerate(folds_idx_):  
            # Plot the roc curve
            fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
            score = average_precision_score(y_.iloc[val_idx], oof_preds_[val_idx])
            scores.append(score)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label='AP fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    precision, recall, thresholds = precision_recall_curve(y_, oof_preds_)
    score = average_precision_score(y_, oof_preds_)
    plt.plot(precision, recall, color='b',
             label='Full ROC (AUC = %0.4f)' % (score),
             lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('{} Recall / Precision'.format(title))
    plt.legend(loc="best")
    plt.tight_layout()
    
    plt.savefig('{}_recall_precision_curve.png'.format(title))
    
def report(test_preds, folds, importances, data, y, oof_preds, title, is_with_nfold):
    print(title)
    test_preds.to_csv('{}_submission.csv'.format(title), index=False)
    # Display a few graphs
    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(data)]
    display_importances(importances, title)
    display_roc_curve(y, oof_preds, folds_idx, title, is_with_nfold)
    display_precision_recall(y, oof_preds, folds_idx, title, is_with_nfold)
    
    
# def opt(features, data_):
#     random.seed(time.clock())
    
#     def lgb_evaluate(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
#         params = {'application':'binary', 'early_stopping_round':100, 'metric':'auc', 'n_estimators':10000, 'nthread' : 6}
#         params["num_leaves"] = int(round(num_leaves))
#         params['feature_fraction'] = max(min(feature_fraction, 1), 0)
#         params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
#         params['max_depth'] = int(round(max_depth))
#         params['lambda_l1'] = max(lambda_l1, 0)
#         params['lambda_l2'] = max(lambda_l2, 0)
#         params['min_split_gain'] = min_split_gain
#         params['min_child_weight'] = min_child_weight        
        
#         us_df = undersample(data_)[0]
#         print('us_df.shape: ',us_df.shape)
#         #categorical_feats = us_df.columns[input_df.dtypes == 'object']
#         train_data = lgb.Dataset(data=us_df[features], label=us_df['TARGET'], free_raw_data=False)
#         random_seed = random.randint(1,100)
#         cv_result = lgb.cv(params, train_data, nfold=5, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
#         print(cv_result)
#         return cv_result

#     def bayesOpt(data_):
#         params = {'num_leaves': (24, 45),
#                 'feature_fraction': (0.1, 0.9),
#                 'bagging_fraction': (0.8, 1),
#                 'max_depth': (5, 8.99),
#                 'lambda_l1': (0, 5),
#                 'lambda_l2': (0, 3),
#                 'min_split_gain': (0.001, 0.1),
#                 'min_child_weight': (5, 50)}
        
#         lgbBO = BayesianOptimization(lgb_evaluate, params)


#         lgbBO.maximize(init_points=5, n_iter=5)
#         best = lgbBO.res['max']
#         print(best)
#         return best


#     return bayesOpt(data_)
def opt(features, data_):
    random.seed(time.clock())
    def bayes_parameter_opt_lgb(data, init_round=15, opt_round=25, n_estimators=10000, learning_rate=0.05, output_process=False):
        
        # parameters
        def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
            params = {'application':'binary','num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':100, 'metric':'auc'}
            params["num_leaves"] = int(round(num_leaves))
            params['feature_fraction'] = max(min(feature_fraction, 1), 0)
            params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
            params['max_depth'] = int(round(max_depth))
            params['lambda_l1'] = max(lambda_l1, 0)
            params['lambda_l2'] = max(lambda_l2, 0)
            params['min_split_gain'] = min_split_gain
            params['min_child_weight'] = min_child_weight
            
            us_df = undersample(data)[0]
            print('us_df.shape: ',us_df.shape)
            #categorical_feats = us_df.columns[input_df.dtypes == 'object']
            train_data = lgb.Dataset(data=us_df[features], label=us_df['TARGET'], free_raw_data=False)
            random_seed = random.randint(1,100)
        
            cv_result = lgb.cv(params, train_data, nfold=3, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
            return max(cv_result['auc-mean'])
        # range 
        lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                                'feature_fraction': (0.1, 0.9),
                                                'bagging_fraction': (0.8, 1),
                                                'max_depth': (5, 8.99),
                                                'lambda_l1': (0, 5),
                                                'lambda_l2': (0, 3),
                                                'min_split_gain': (0.001, 0.1),
                                                'min_child_weight': (5, 50)}, random_state=0)
        # optimize
        lgbBO.maximize(init_points=init_round, n_iter=opt_round)

        # output optimization process
        if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")

        # return best parameters
        return lgbBO.res['max']['max_params']

    opt_params = bayes_parameter_opt_lgb(data_, init_round=5, opt_round=10, n_estimators=100, learning_rate=0.05)




