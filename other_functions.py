import pandas as pd
import numpy as np

from lag_features import *
from imputing_functions import *

def apply_calc(df_,calculations):
    for key, value in calculations.items():
        if 'gb_list' in value:
            gb_list = value['gb_list']
        else:
            gb_list = None
        
        if 'funs' in value:
            funs = value['funs']
        else:
            funs = None
            
        if 'windows' in value:
            windows = value['windows']
        else:
            windows = None
        
        target = value['target']
        shifts = value['shifts']
        
        if 'fillna_strat' in value:
            fillna_strat = value['fillna_strat']
        else:
            fillna_strat = None
        
        if windows is None:
            if funs is None:
                for shift in shifts:
                    df_ = TS_Feature_Generator(df=df_.copy(), gb_list=gb_list, target=target, shift=shift, 
                                            fillna_strategy=fillna_strat)
            else:
                for fun in funs:
                    for shift in shifts:
                        df_ = TS_Feature_Generator(df=df_.copy(), gb_list=gb_list, target=target, shift=shift, fun=fun, 
                                                fillna_strategy=fillna_strat)
        else:
            for fun in funs:
                for window in windows:
                    for shift in shifts:
                        df_ = TS_Feature_Generator(df=df_.copy(), gb_list=gb_list, target=target, 
                                                shift=shift, window=window, fun=fun, fillna_strategy=fillna_strat)
                                                
    return df_


def date_features(df, col):
    df[col] = pd.to_datetime(df[col])
    df['month'] = df[col].dt.month
    df['day'] = df[col].dt.day
    df['year'] = df[col].dt.year

    df['year_month'] = [ str(y)+'_'+str(m) if m < 10 else str(y)+'_0'+str(m) for y, m in zip(df['year'], df['month']) ]
    df['weekday'] = df[col].dt.weekday

    return df


def delete_time_features(df):
    cols = ['month','year','day','weekday','weekday_type']
    for col in cols:
        del df[col]
        
    return(df)