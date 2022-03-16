from datetime import datetime
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import shap
import matplotlib.pyplot as plt


def _xgb(df,target,pred_func,pipeline_name,hyperparam_search=False,shap=False):
    
    """Run XGBoostRegressor using specified features.
    Parameters
    ----------
    df : pandas.dataframe
        dataframe of features with shape (n_subjects, n_features).
    pred_func: function()
        function that pulls predictors from df
    pipeline_name: string
        name of pipeline
    hyperparam_search: string or bool
        Should we perform hyperparameter tuning on model. Valid strings 'grid' for grid search
        or 'bayes' for bayes search
    shap: bool
        display shap value plots of xgboost model?

    Returns
    -------
    row : pd.Series
        Series including the type of model, pipeline name, and train/test scores
    """
    get_preds = FunctionTransformer(pred_func)
    # pipe_dict = {'demo':FunctionTransformer(get_demos),
    #             'wm':FunctionTransformer(get_wm),
    #             'wm_demo':FunctionTransformer(get_wm_demos),
    #             'demo_no_age':FunctionTransformer(get_demos_no_age)}
    
    start = datetime.now()
    print(target+": ")
    
    if np.isnan(df[target]).any():
        print('excluding subjects with missing behavoiral information:', np.array(df['subjectID'])[np.where(np.isnan(df[target]))])
        y = df[target][np.isnan(df[target]) == False]
        X = df[np.isnan(df[target]) == False]
    
    else:
        X=df
        y=df[target]
        
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)
                                                                      
    if hyperparam_search=='bayes':
        
        xgb_reg = XGBRegressor(
            objective="reg:squarederror",
            nthread=-1
        )
        
        params = {
            "n_estimators": (50, 2001),
            "min_child_weight": (1, 11),
            "gamma": (0.01, 5.0, "log-uniform"),
            "eta": (0.005, 0.5, "log-uniform"),
            "subsample": (0.2, 1.0),
            "colsample_bytree": (0.2, 1.0),
            "max_depth": (2, 6),
        }

        opt = BayesSearchCV(
            xgb_reg,
            params,
            n_iter=100,
            verbose=2
        )
        
        xgb_steps = [
        # ('select',pipe_dict[pipeline]),
        ('select',get_preds),
        # ('impute',SimpleImputer(missing_values=np.nan, strategy='median')),
        ('scale', StandardScaler()),
        ('estimate',opt)]

        xgb_pipe = Pipeline(xgb_steps,verbose=True)

        _ = xgb_pipe.fit(X_train, y_train)

        train_score = xgb_pipe.score(X_train, y_train)
        test_score = xgb_pipe.score(X_test, y_test)

        test_preds = xgb_pipe.predict(X_test)
        train_preds = xgb_pipe.predict(X_train)

        # set the best estimator as xgb_reg to get feature imporance
        xgb_reg = opt.best_estimator_
        
    elif hyperparam_search=='grid':
        
        xgb_reg = XGBRegressor(
            objective="reg:squarederror",
            nthread=-1
        )
        
        xgb_params = {
        'min_child_weight': [2, 3, 5],
        'max_depth': [2, 3, 4],
        'learning_rate':[.005],
        'n_estimators':[1000,2000,3000]
        }
    
        opt = GridSearchCV(xgb_reg,xgb_params,verbose=2,cv=5)

        
        xgb_steps = [
        # ('select',pipe_dict[pipeline]),
        ('select',get_preds),
        # ('impute',SimpleImputer(missing_values=np.nan, strategy='median')),
        ('scale', StandardScaler()),
        ('estimate',opt)]

        xgb_pipe = Pipeline(xgb_steps,verbose=True)

        _ = xgb_pipe.fit(X_train, y_train)

        train_score = xgb_pipe.score(X_train, y_train)
        test_score = xgb_pipe.score(X_test, y_test)

        test_preds = xgb_pipe.predict(X_test)
        train_preds = xgb_pipe.predict(X_train)

        # set the best estimator as xgb_reg to get feature imporance
        xgb_reg = opt.best_estimator_
    else:
        xgb_reg = XGBRegressor(
            objective="reg:squarederror",
            nthread=-1,
            colsample_bytree=0.86,
            gamma=0.01,
            learning_rate=0.005,
            max_depth=2,
            min_child_weight=2,
            n_estimators=3000,
            subsample=0.2
        )
        
        xgb_steps = [
        # ('select',pipe_dict[pipeline]),
        ('select',get_preds),
        # ('impute',SimpleImputer(missing_values=np.nan, strategy='median')),
        ('scale', StandardScaler()),
        ('estimate',xgb_reg)]

        xgb_pipe = Pipeline(xgb_steps,verbose=True)

        _ = xgb_pipe.fit(X_train, y_train)

        train_score = xgb_pipe.score(X_train, y_train)
        test_score = xgb_pipe.score(X_test, y_test)

        test_preds = xgb_pipe.predict(X_test)
        train_preds = xgb_pipe.predict(X_train)


    print('Elapsed time: ', datetime.now() - start)
    
    # get information on the XGBRegressor
    print(xgb_reg)
#     print(xgb_reg.importance_type)
        
    # feature importance
    import xgboost
    
    # could plot all features by looking for non-zero feature_importances_
    num_features = 30

   
    mod = xgb_pipe.named_steps['estimate']

 
    if shap:
        import shap
        
        # shap_df = pipe_dict[pipeline].fit_transform(df)
        shap_df = get_preds.fit_transform(df)
        if hyperparam_search:
            explainer = shap.explainers.Tree(xgb_pipe.named_steps['estimate'].best_estimator_)
        else:
            explainer = shap.explainers.Tree(xgb_pipe.named_steps['estimate'])
            
        shap_values = explainer(shap_df)

        shap.summary_plot(shap_values.values, shap_df,show=False)
        plt.show()

        # test
#         shap_values = explainer(X_test)
        shap.plots.bar(shap_values, max_display=num_features)
        plt.show()

        # To get an overview of which features are most important for a model
        # plot the SHAP values of every feature for every sample. The plot
        # sorts features by the sum of SHAP value magnitudes over all samples,
        # and uses SHAP values to show the distribution of the impacts each
        # feature has on the model output. The color represents the feature
        # value (red high, blue low).
        plt.show()


    return mod, pd.Series(data=['xgb',target, pipeline_name, train_score, test_score], 
                     index=['model','target','pipeline', 'train score', 'test score']), test_preds, y_test