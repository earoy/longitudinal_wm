import seaborn as sns
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import os.path as op
sys.path.insert(0,'../utils')
from read_math_utils_ping import *
from xgb_util import _xgb
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV
from skopt.space import Real, Categorical, Integer
import shap

# Load data
combo_df = pd.read_csv('../data/final_df_ping.csv')

scores = []
# pipe_dict = {'demo':get_demos}
# target_list = ['nihtbx_reading_agecorrected']
pipe_dict = {'demo':get_demos,'wm':get_wm,'wm_demo':get_wm_demos}
target_list = ['nihtbx_reading_agecorrected','interview_age']

for pipe in pipe_dict.keys():
    for target in target_list:
        preds_df = pd.DataFrame()
        if target == 'interview_age' and pipe == 'demo':
            mod,mod_scores,mod_preds,test = _xgb(combo_df,target,get_demos_no_age,pipe,hyperparam_search='grid',shap=True)
            scores.append(mod_scores)
            
        elif target == 'interview_age' and pipe == 'wm_demo':
            mod,mod_scores,mod_preds,test = _xgb(combo_df,target,get_wm_no_age_demos,pipe,hyperparam_search='grid',shap=True)
            scores.append(mod_scores)
            
        else:
            mod,mod_scores,mod_preds,test = _xgb(combo_df,target,pipe_dict[pipe],pipe,hyperparam_search='grid',shap=True)
            scores.append(mod_scores)
        
        file_name = target + '_'+ pipe + '_preds_ping.csv'
        preds_df['pred'] = mod_preds.tolist()
        preds_df['obs'] = test.tolist()
        preds_df.to_csv(file_name,index=False)


pd.DataFrame(scores).to_csv('ping_scores.csv',index=False)
# pd.Series(read_wm_preds).to_csv('read_wm_default_preds_abcd_t1.csv',index=False)

