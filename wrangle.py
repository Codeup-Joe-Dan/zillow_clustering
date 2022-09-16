import os
import pandas as pd
import numpy as np
import env
from sklearn.model_selection import train_test_split
from datetime import datetime

def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''
    function to generate a url for querying the codeup database
    accepts a database name (string) and requires an env.py file with 
    username, host, and password.

    Returns an url as a string  
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


def get_zillow():
    """
    Retrieve locally cached data .csv file for the zillow dataset
    If no locally cached file is present retrieve the data from the codeup database server
    Keyword arguments: none
    Returns: DataFrame

    """
    
    filename = "zillow.csv"

    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    else:
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use 
        df = pd.read_sql('''
                SELECT parcelid, bathroomcnt as bathrooms, bedroomcnt as bedrooms,
                                    calculatedfinishedsquarefeet as sqft, 
                                    fips as county, fullbathcnt, latitude, garagecarcnt, garagetotalsqft as garagesqft,
                                    longitude, lotsizesquarefeet as lotsize, poolcnt, fireplacecnt,
                                    rawcensustractandblock as tract, regionidzip, yearbuilt, 
                                    structuretaxvaluedollarcnt as structuretaxvalue, propertylandusedesc,
                                    taxvaluedollarcnt as taxvalue, landtaxvaluedollarcnt as landtaxvalue,
                                    taxamount, logerror
                            FROM properties_2017
                            JOIN predictions_2017
                            USING (parcelid)
                            LEFT JOIN propertylandusetype
                            USING (propertylandusetypeid)
                            HAVING propertylandusedesc = 'Single Family Residential'
                            ORDER BY transactiondate
                    ''', get_connection('zillow'))

    # Remove multiple instances of property, keeping the latest transaction (query was ordered by transaction date)
    df = df[~df.duplicated(subset=['parcelid'],keep='last')]

    # Write that dataframe to disk for later. This cached file will prevent repeated large queries to the database server.
    df.to_csv(filename, index=False)
    return df

def prep_zillow(df):
    '''
    function accepts a dataframe of zillow data and prepares it for use in 
    modelling
    
    returns a transformed dataframe with features for exploration and modeling
    '''


    # create column with fips value converted from an integer to the county name string
    df['county'] = df.county.map({6037 : 'Los Angeles', 6059 : 'Orange', 6111 : 'Ventura'})

    # drop rows with null in tract column, trim FIPS and decimal portion, convert to int
    df = df[df['tract'].notna()]
    df.tract = df.tract.astype(str).str[4:8]
    df.tract = df.tract.astype(int)

    # convert fireplace count nulls to 0
    df.fireplacecnt = df.fireplacecnt.fillna(0)
    # garage null values to 0
    df.garagecarcnt = df.garagecarcnt.fillna(0)
    df.garagesqft = df.garagesqft.fillna(0)

    # convert poolcnt nulls to 0's
    df.poolcnt = df.poolcnt.fillna(0)

    # convert lat and long to proper decimal format
    df['latitude'] = df['latitude'] / 1_000_000
    df['longitude'] = df['longitude'] / 1_000_000

    #change year to age
    df['age'] = 2022 - df['yearbuilt'] 
    df.drop(columns=['yearbuilt'], inplace=True)

    # drop remaining nulls
    df = df.dropna()

    # drop 0 bedroom rows
    df = df[df.bedrooms > 0]

    # drop rows with zip over 99999
    df= df[df.regionidzip <= 99999]

    # drop property use type that is no longer needed
    df.drop(columns=['propertylandusedesc'], inplace=True)

    # create absolute error column
    df['abserror'] = abs(df.logerror)
    
    # one-hot encode county
    dummies = pd.get_dummies(df['county'],drop_first=False)
    df = pd.concat([df, dummies], axis=1)


    return df

    
def remove_outliers(df):
    ''' remove outliers from a list of columns in a dataframe 
        accepts a dataframe 

        returns a dataframe with outliers removed
    '''
    col_list = ['bathrooms', 'bedrooms', 'sqft', 'fullbathcnt',
       'latitude', 'longitude', 'lotsize',
       'structuretaxvalue', 'taxvalue', 'landtaxvalue', 'taxamount']
    
    for col in col_list:

        q1, q3 = df[col].quantile([.13, .87]) # get range
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + 1.5 * iqr   # get upper bound
        lower_bound = q1 - 1.5 * iqr   # get lower bound

        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]

    return df

def my_split(df):
       '''
       Separates a dataframe into train, validate, and test datasets

       Keyword arguments:
       df: a dataframe containing multiple rows
       

       Returns:
       three dataframes who's length is 60%, 20%, and 20% of the length of the original dataframe       
       '''

       # separate into 80% train/validate and test data
       train_validate, test = train_test_split(df, test_size=.2, random_state=333)

       # further separate the train/validate data into train and validate
       train, validate = train_test_split(train_validate, 
                                          test_size=.25, 
                                          random_state=333)

       return train, validate, test

def wrangle_zillow():
    df = get_zillow() # get the data
    df = prep_zillow(df) # prep the data
    df = remove_outliers(df) # remove outliers

    return my_split(df)