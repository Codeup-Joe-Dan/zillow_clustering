import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from IPython.display import display_html 
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor



def predict_baseline(train):
    '''
    Function to calculate the RMSE for the mean and median logerror of zillow properties
    accepts train dataframe, displays a table of formatted results, and returns a results table
    '''

    # create y_train and y_validate
    y_train = train['logerror']
        
    y_train = pd.DataFrame(y_train)
    
    value_pred_mean = y_train['logerror'].mean()
    y_train['value_pred_mean'] = value_pred_mean

    # compute value_pred_median
    value_pred_median = y_train['logerror'].median()
    y_train['value_pred_median'] = value_pred_median

    # RMSE of value_pred_mean
    rmse_train = mean_squared_error(y_train.logerror, y_train.value_pred_mean)**(1/2)

    results = pd.DataFrame(columns = ['model', 'RMSE_train', 'RMSE_validate', 'R2'])
    newresult = ['Mean','{:,.4f}'.format(rmse_train),'N/A', 'N/A']
    results.loc[len(results)] = newresult

    # RMSE of value_pred_median
    rmse_train = mean_squared_error(y_train.logerror, y_train.value_pred_median)**(1/2)

    # create and display tabular formatted data
    newresult = ['Median','{:,.4f}'.format(rmse_train),'N/A', 'N/A']
    results.loc[len(results)] = newresult

    df_style = results.style.set_table_attributes("style='display:inline; margin-right:100px;'").set_caption("RESULTS")
    display_html(df_style._repr_html_(), raw=True)

    return newresult