
import pandas as pd

def extract_features(df, features_list):
    
    feature_df = df.filter(regex='|'.join(features_list))
    return feature_df

def get_reading(df):
    return df['nihtbx_reading_agecorrected']


def get_age(df):
    return df['interview_age']

def get_demos(df):
    # df = df.drop(columns={'scan_site_id'})
    return extract_features(df, ['interview_age','demo_comb_income_v2'])

def get_demos_no_age(df):
    # df = df.drop(columns={'scan_site_id'})
    return extract_features(df, ['demo_comb_income_v2'])

def get_qc_demos(df):
    # df = df.drop(columns={'scan_site_id'})
    return extract_features(df, ['Age','demo_comb_income_v2','qc_score'])

def get_qc_demos_no_age(df):
    # df = df.drop(columns={'scan_site_id'})
    return extract_features(df, ['demo_comb_income_v2','qc_score'])

def get_wm(df):
    return extract_features(df, ['dmdtifp1'])

def get_wm_demos(df):
    # df = df.drop(columns={'scan_site_id'})
    return extract_features(df, ['dmdtifp1','interview_age','demo_comb_income_v2'])

def get_wm_no_age_demos(df):
    # df = df.drop(columns={'scan_site_id'})

    return extract_features(df, ['dmdtifp1','demo_comb_income_v2'])

def merge_behavioral(tract_profiles, behavioral_data,identifier,measures):
    #Merge behavioral data with pyAFQ csv output based on a subject ID identifier
    #In the case of HBN behavioral data EID == subjectID
    # measures should be a list of whatever behvaioral metrics you want
    
    cols = measures + ['subjectID']
    
    behavioral_data = behavioral_data.rename(columns = {identifier : 'subjectID'})
    behavioral_data_filtered = behavioral_data[cols]
    
    merged = pd.merge(behavioral_data_filtered, tract_profiles, on=['subjectID'])
    
    return merged

def combine_target_feature_df(diffusion_metrics,subjects,behavioral_metrics,subjectID_behavioral,features_list,feature_names):
    
    diffusion_df = pd.DataFrame(diffusion_metrics)
    diffusion_df.columns = ['_'.join(feature_names[i][0:2]+(str(i%100),)) if 0 <= i < len(feature_names) else x 
                for i, x in enumerate(diffusion_df.columns, 0)]
    
    diffusion_df['subjectID'] = subjects
    
    behavioral_df = pd.DataFrame(behavioral_metrics)
    features_list.append('subjectID')
        
    behavioral_df = behavioral_df.rename(columns = {subjectID_behavioral : 'subjectID'})
    

    combined_df = pd.merge(diffusion_df,behavioral_df[features_list], left_on="subjectID",right_on="subjectID", how="inner")
    
    return combined_df