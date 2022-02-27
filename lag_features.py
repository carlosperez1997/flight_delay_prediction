from imputing_functions import *
import numpy as np

def function_map(fun):
    if fun == 'max':
        return(np.max)
    elif fun == 'min':
        return(np.min)
    elif fun == 'std':
        return(np.std)
    elif fun == 'mean':
        return(np.mean)
    elif fun == 'median':
        return(np.median)
    
    
def TS_shift_woutfun_FG(df, target, shift):
    name_ = target+'_s'+str(shift)
    try:
        df[name_] = df[target].transform(lambda x: x.shift(shift))
    except:
        df[name_] = df[target].apply(lambda x: x.shift(shift))
    print('Generating',name_)
    return(df, name_)


def TS_shift_wfun_FG(df, target, shift, fun):
    func = function_map(fun)
    name_ = target+'_s'+str(shift)+'_'+fun
    try:
        df[name_] = df[target].transform(lambda x: x.shift(shift).expanding().apply(func))
    except:
        df[name_] = df[target].apply(lambda x: x.shift(shift).expanding().apply(func))
    print('Generating',name_)
    return(df, name_)


def TS_shift_rolling_FG(df, gb_list, target, shift, window, fun):
    func = function_map(fun)
    name_ =  target+'_s'+str(shift)+'_r'+str(window)+'_'+fun
    try:
        df[name_] = df[target].transform(lambda x: x.shift(shift).rolling(window).apply(func))
    except:
        df[name_] = df[target].apply(lambda x: x.shift(shift).rolling(window).apply(func))
    print('Generating',name_)
    return(df, name_)


def TS_gb_shift_woutfun_FG(df, gb_list, target, shift):
    name_ = target+'_'+'_'.join(gb_list)+'_s'+str(shift)
    try:
        df[name_] = df.groupby(gb_list)[target].transform(lambda x: x.shift(shift))
    except:
        df[name_] = df.groupby(gb_list)[target].apply(lambda x: x.shift(shift))
    print('Generating',name_)
    return(df, name_)


def TS_gb_shift_wfun_FG(df, gb_list, target, shift, fun):
    func = function_map(fun)
    name_ = target+'_'+'_'.join(gb_list)+'_s'+str(shift)+'_'+fun
    try:
        df[name_] = df.groupby(gb_list)[target].transform(lambda x: x.shift(shift).expanding().apply(func))
    except:
        df[name_] = df.groupby(gb_list)[target].apply(lambda x: x.shift(shift).expanding().apply(func))
    print('Generating',name_)
    return(df, name_)


def TS_gb_shift_rolling_FG(df, gb_list, target, shift, window, fun):
    func = function_map(fun)
    name_ = target+'_'+'_'.join(gb_list)+'_s'+str(shift)+'_r'+str(window)+'_'+fun
    try:
        df[name_] = df.groupby(gb_list)[target].transform(lambda x: x.shift(shift).rolling(window).apply(func))
    except:
        df[name_] = df.groupby(gb_list)[target].apply(lambda x: x.shift(shift).rolling(window).apply(func))
    print('Generating',name_)
    return(df, name_)


def TS_Feature_Generator_processing(df, target, shift, gb_list = None, window = None, fun = None):
    if gb_list is None:
        if window is None:
            if fun is None:
                df, name_ = TS_shift_woutfun_FG(df, target, shift)
            else:
                df, name_ = TS_shift_wfun_FG(df, target, shift, fun)
        else:
            df, name_ = TS_shift_rolling_FG(df, target, shift, window, fun)
    else:
        if window is None:
            if fun is None:
                df, name_ = TS_gb_shift_woutfun_FG(df, gb_list, target, shift)
            else:
                df, name_ = TS_gb_shift_wfun_FG(df, gb_list, target, shift, fun)
        else:
            df, name_ = TS_gb_shift_rolling_FG(df, gb_list, target, shift, window, fun)
            
    return(df, name_)


def TS_Feature_Generator(df, target, shift, gb_list = None, window = None, fun = None, fillna_strategy = None):
    
    # Time Series Feature Generator
    df, name_ = TS_Feature_Generator_processing(df, target, shift, gb_list, window, fun)
            
    # Imputing (fill na strategy)
    if fillna_strategy is not None:
        df = Imputing_Functions(df, name_, gb_list, fillna_strategy)
            
    return(df)