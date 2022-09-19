# Joe Bennett & Dan Churchill - Predicting Zillow Log Error using Clustering and Linear Regression


# Goals

To be able to predict the log error of Zestimates for single family homes that were sold in 2017 using regression algorithms and lend insight to the data science team on what factors drive the log error.






# Description

The Zillow data science team has requested our help developing a model that uses property attributes of single family houses sold in 2017 to predict the log error produced by their model. As the data scientists tasked with this we will use what we learn to help the Zillow data science team improve their existing predictive model.



# Planning

Data science across all domains can usually be generalized as the following steps. We used this as a framework for making our plan.

Planning- writing out a timeline of actionable items, when the MVP will be finished and how to know when it is, formulate initial questions to ask the data.

Acquisition- Gather our data and bring all necessary data into our python enviroment from the codeup SQL server 

Preparation- this is blended with acquisition where we clean and tidy the data and split into 60% train, 20% validate, and 20% test 

Exploration/Pre-processing- where we will create visualizations and conduct hypothesis testing to select and engineer features that impact the target variable.  This includes clustering, where we will combined features to try and find meaningful insights.

Modeling- based on what we learn in the exploration of the data we will select the useful features and feed them into different regression models and evaluate performance of each to select our best perfomoing model.

Delivery- create a final report that succintly summarizes what we did, why we did it and what we learned in order to make recommendations


# Initial hypothesis of statistical testing

## 1 - Is the Log Error different by County?
H_null= Average Log Error of the properties in three counties (Los Angeles, Ventura, and Orange) are all equal.

H_a= Average Log Error of the properties in three counties (Los Angeles, Ventura, and Orange) are NOT all equal

We rejected the null hypothesis

## 2 - Does Lot Size impact log error?
H_null= Log Error of small lots is equal to Log Error of all properties

H_a= Log Error of small lots is significantly different than Log Error of all properties 

We rejected the null hypothesis

# Data dictionary 

| Feature | Definition | Data Type |
| ----- | ----- | ----- |
| parcelid | Unique id for each property| int |
| bathrooms| The number of bathrooms on the property | float |
| bedrooms | The number of bedrooms on the property | float |
| sqft | the square footage of the land | float |
| county| The County the property is located in | string |
| latitude | The geographical latitude of the property | float |
| longitude | The geographical longitude of the property | float |
| garagesqft | The area of the garage in sq ft | float |
| lotsize | the square footage of the land | float |
| regionidzip | the zipcode of the property | float |
| structuretaxvalue | the value of the building on the property | float |
| landtaxvalue | the value of the land | float |
| logerror | the logerror of zillow's model | float |
| age | the age of the property in years | float |
| abserror | the absolute value of the logerror | float |
| dollarspersqft | the assessed value per sq ft of land area | float |
| Los Angeles | 1 if the property is in LA County | int |
| Orange | 1 if the property is in Orange County | int |
| Ventura | 1 if the property is in Ventura County | int |




# how to reproduce our work

To reproduce our work you will need your own env.py file in the same directory with your credentials ( host, user, password).  You will also need to have our explore.py, wrangle.py and modeling.py files in the local directory for functions to work properly. With that our report will run completely from the top down. 








# key findings and recommendations

We looked at location features and concluded that they did have an impact on logerror.  We also tested lotsize and found that it too had correlation to log error.  We also tested several interations of clustering features, settling on clusters that grouped lot size and taxamount. Our best model was an OLS model that performed slightly better on training data than on validate data.  This continued on test data, indicating our model is likely overfit.  We were not able to improve upon baseline (mean) RMSE with our model.  

If time allowed we would like to re evaluate the features in order to better engineer a model that can tilize the available data while avoiding overfitting.  The clusters we utilized with lot size and tax amount were significant, but further insight was likely available in bedrooms, bathrooms, and regional data.

