import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import gc

def get_train_test_label(row=None):
    train = pd.read_csv("main_table.csv", nrows = row)
    test = pd.read_csv("test_table.csv", nrows = row)
    label = train['TARGET']
    train = train.drop(columns=['TARGET','Unnamed: 0'])
    test = test.drop(columns=['TARGET','Unnamed: 0'])
    #train = train.iloc[:,0:-50]
    #test = test.iloc[:,0:-50]
    #feats = [f for f in test.columns if f not in train.columns]
    #print(feats)
    print(train.shape)
    print(test.shape)
    
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

def cross_validation(data_, test_, y_, model):
    folds_ = KFold(n_splits=5, shuffle=True, random_state=546789)
    
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    
    feature_importance_df = pd.DataFrame()
    
    feats = [f for f in data_.columns if f not in ['SK_ID_CURR']]
    
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
    test_[['SK_ID_CURR', 'TARGET']].to_csv('{}_submission.csv'.format(model), index=False, float_format='%.8f')
    return oof_preds, test_[['SK_ID_CURR', 'TARGET']], feature_importance_df, folds_

def display_importances(feature_importance_df_,title):
    # Plot feature importances
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index
    print(cols)
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    print(best_features)
    plt.figure(figsize=(8,10))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('{} Features (avg over folds)'.format(title))
    plt.tight_layout()
    plt.savefig('{}_importances.png'.format(title))
    
def display_roc_curve(y_, oof_preds_, folds_idx_,title):
    # Plot ROC curves
    plt.figure(figsize=(6,6))
    scores = [] 
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
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC Curve'.format(title))
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    plt.savefig('{}_roc_curve.png'.format(title))
    
def display_precision_recall(y_, oof_preds_, folds_idx_, title):
    # Plot ROC curves
    plt.figure(figsize=(6,6))
    
    scores = [] 
    for n_fold, (_, val_idx) in enumerate(folds_idx_):  
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = average_precision_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='AP fold %d (AUC = %0.4f)' % (n_fold + 1, score))
    
    precision, recall, thresholds = precision_recall_curve(y_, oof_preds_)
    score = average_precision_score(y_, oof_preds_)
    plt.plot(precision, recall, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('{} Recall / Precision'.format(title))
    plt.legend(loc="best")
    plt.tight_layout()
    
    plt.savefig('{}_recall_precision_curve.png'.format(title))
    
def report(test_preds, folds, importances, data, y, oof_preds, title):
    print(title)
    test_preds.to_csv('{}_submission.csv'.format(title), index=False)
    # Display a few graphs
    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(data)]
    display_importances(importances, title)
    display_roc_curve(y, oof_preds, folds_idx, title)
    display_precision_recall(y, oof_preds, folds_idx, title)




