
import pandas as pd

def extract_features(df, features_list):
    
    feature_df = df.filter(regex='|'.join(features_list))
    return feature_df

def get_reading(df):
    return df['zscore']

def get_reading_raw(df):
    return df['tbx_reading_score']

def get_reading_bin(df):
    return df['reading_score_bin']

def get_demos(df):
    return extract_features(df, ['interview_age','income'])

def get_demos_no_age(df):
    return extract_features(df, ['income'])

def get_age(df):
    return extract_features(df, ['interview_age'])

def get_wm(df):
    return extract_features(df, ['dti_fiber'])


def get_wm_demos(df):
    return extract_features(df, ['dti_fiber','interview_age','income'])

def get_wm_ses(df):
    return extract_features(df, ['dti_fiber','income'])

def get_wm_no_age_demos(df):

    return extract_features(df, ['dti_fiber','income'])
