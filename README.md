# Goals

To be able to predict assessed value of of single family homes that were sold in 2017 using regression algorithms and lend insight to the data science team on how Zillow can improve their existing model- what works and what does not. 






# Description

the Zillow data science team has requested my help developing a model that uses property attributes of single family houses sold in 2017 to predict their assessed value. As the data scientist tasked with this I will use what I learn to help the Zillow data science team improve the existing predictive model.



# Planning

data science across all domains can usually be generalized as the following steps. I use this as a framework for making my plan.

Planning- writing out a timeline of actionable items, when the MVP will be finished and how to know when it is, formulate initial questions to ask the data.

Acquisition- Gather my data and bring all necessary data into my python enviroment from SQL server 

Preparation- this is blended with acquisition where I will clean and tidy the data and split into my train, validate, and test 

Exploration/Pre-processing- where i will create visualizations and conduct hypothesis testing to select and engineer features that impact the target variable.

Modeling- based on what i learn in the exploration of the data I will select the useful features and feed into different regression models and evaluate performance of each to select my best perfomoing model.

Delivery- create a final report that succintly summarizes what I did, why I did it and what I learned in order to make recommendations


# Initial hypothesis

## 1
H_null= there is no correlation between number of bedrooms and assessed value
H_a= there is a linear relationship between number of bedrooms and assessed value

## 2 
H_null= there is no relationship between number of bathrooms and assessed value
H_a= there is a linear relationship between number of bathrooms and assessed value

## 3
H_null= there is no relationship between livable square feet and assessed value
H_a= there is a linear relationship between livable square feet and assessed value

## 4
H_null= there is no relationship between non-bed & non-bath sq ft and assessed value
H_a= there is a linear relationship between non-bed & non-bath sq ft and assessed value

# Data dictionary 

column name                             description

bedrooms                               the number of bedrooms in the property

bathrooms                              the number of bathrooms in the property

area                                   the total square livable footage of the property

lot_area                               the total square footage of the parcel/lot the property is on

tax_value                              the assessed value imposed by the local taxing authority

year_built	                           the year the home was constructed

fips	                               the digit code representing what county the property is in

lot_living_ratio	                   the ratio of the size of the square footage of the house in relationship to the lot sq footage

non_bed_bath_area                      the total square footage of the house minus the square footage of bedrooms and bathrooms
                                       average room sizes for that area were used then multiplied by the corresponding room count





# how to reproduce my work

To reproduce my work you will need your own env.py file in the same directory with your credentials ( host, user, password). you will also need to have my wrangle.py and evaluate.py files in the local directory for functions to work properly. with that my report will run completely from the top down. 








# key findings and recommendations

I looked at number of bedrooms, bathrooms, square footage, and square footage not included in bed or bath as features and confirmed they all have a slight statistically significant positive linear correlation. My best model performed consistently across data sets and even slightly improved in the final test data set. I expect this model will perform similarly on future data.

If you notice the graphs of errors in predicions there is unmistakable downward trend compared to actual values. This suggests there are other features not yet captured in my models driving that trend. My highest correlated feature to assessed value was the one I engineered.  For a business perspective I recommend incorporating my desined features into zillow's current model to increase its accuracy. 

If time allowed I would like to re evaluate the zillow database for possible appropriate features to improve my model's accuracy that are not derived from square footage of the house, do some feature engineering, and rerun the predictions. While my top model surely beats the baseline I feel it is far from a huge success. much improvement is possible.

