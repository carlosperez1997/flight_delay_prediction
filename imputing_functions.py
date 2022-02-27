import pandas as pd 
import numpy as np 
from sklearn.impute import KNNImputer

def ConstantImputer(X, target, value=None):
    if value is None:
        value = -1
    print(' - Constant Imputing', target)
    return X.fillna(-1)


def GroupMeanImputer(X, group_var, target):
    X_ = X.copy()
    gp_summ = X_.groupby(group_var)[target].mean().reset_index()
    cols = group_var.copy()
    cols.append(target)
    X_ = X[cols].merge(gp_summ, on=group_var, how='left')
    X.loc[ X[target].isnull(), target] = X_.loc[ X[target].isnull(), target+'_y']
    print(' - Mean Imputing', target, 'by', group_var)
    return X


def GroupMinImputer(X, group_var, target):
    X_ = X.copy()
    gp_summ = X_.groupby(group_var)[target].min().reset_index()
    cols = group_var.copy()
    cols.append(target)
    X_ = X[cols].merge(gp_summ, on=group_var, how='left')
    X.loc[ X[target].isnull(), target] = X_.loc[ X[target].isnull(), target+'_y']
    print(' - Min Imputing', target, 'by', group_var)
    return X
    

def GroupMedianImputer(X, group_var, target):
    X_ = X.copy()
    gp_summ = X_.groupby(group_var)[target].median().reset_index()
    cols = group_var.copy()
    cols.append(target)
    X_ = X[cols].merge(gp_summ, on=group_var, how='left')
    X.loc[ X[target].isnull(), target] = X_.loc[ X[target].isnull(), target+'_y']
    print(' - Median Imputing', target, 'by', group_var)
    return X


def customKNNImputer(X, target, n_neighbors = 3, max_samples = None):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    X = X.copy(deep=True)
    X_ = X[num_cols]
    if max_samples is not None:
        X_ = X_.sample(max_samples)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_ = pd.DataFrame(imputer.fit_transform(X_), columns = num_cols)
    X.loc[ X[target].isnull(), target] = X_.loc[ X[target].isnull(), target ]
    print(' - KNN Imputing', target)
    return X


def Imputing_Functions(df, name_, gb_list, fillna_strategy):
    if fillna_strategy == 'bfill':
        df[name_] = df.groupby(gb_list)[name_].bfill()
        print(' - Backward fill ',name_)

    elif fillna_strategy == 'ffill':
        df[name_] = df.groupby(gb_list)[name_].ffill()
        print(' - Forward fill ',name_)

    elif fillna_strategy[0] == 'MeanGroupImputer':
        group_var = fillna_strategy[1]
        df = GroupMeanImputer(df, group_var, name_)
            
    elif fillna_strategy[0] == 'MinGroupImputer':
        group_var = fillna_strategy[1]
        df = GroupMinImputer(df, group_var, name_)
            
    elif fillna_strategy[0] == 'MedianGroupImputer':
        group_var = fillna_strategy[1]
        df = GroupMedianImputer(df, group_var, name_)
    
    elif fillna_strategy[0] == 'KNNImputer':
        if len(fillna_strategy) == 2:
            n_neighbors = int(fillna_strategy[1])
        else:
            n_neighbors = None
        df = customKNNImputer(df, name_, n_neighbors=n_neighbors)
    
    return df