# Energy Forecasting: Data Mining & Predictive Analysis

## Project Overview
This repository explores various data mining and machine learning approaches towards predicting, classifying, and understanding the relationships and patterns in global energy consumption and CO2 emissions. Leveraging a rich dataset, we delve deep into Random Forest Regression, Decision Trees, Clustering, Association Rule Mining, and Time Series Forecasting to unearth insights and forecasts pertinent to World Energy dynamics.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Methodologies Applied](#methodologies-applied)
   - [Random Forest Regression Analysis](#random-forest-regression-analysis)
   - [Classification: Decision Tree & Random Forest](#classification-decision-tree--random-forest)
   - [Association Rule Mining: Apriori Algorithm](#association-rule-mining-apriori-algorithm)
   - [Clustering: KMeans Algorithm](#clustering-kmeans-algorithm)
   - [Regression Analysis: XGBoost](#regression-analysis-xgboost)
   - [Time Series Forecasting: ARIMA](#time-series-forecasting-arima)
4. [Results & Insights](#results--insights)
5. [Conclusions](#conclusions)
6. [Setup & Requirements](#setup--requirements)

## Introduction
The global challenge of reducing CO2 emissions and adapting to a sustainable energy paradigm makes it imperative to understand the patterns, associations, and forecasting of energy consumption and emissions. This project employs various data mining techniques to dissect and predict energy-related variables, with a particular focus on CO2 emissions across different continents.

## Data Loading
We began by loading our dataset, which contains informa􀆟on about CO2
emissions, energy consump􀆟on, produc􀆟on, GDP, popula􀆟on, energy intensity per capita
and energy intensity by GDP for different countries from 1980 to 2019.

## Data Preprocessing
First, we checked the data for duplicates to avoid biasing the
performance of the analysis and machine learning model by giving more weight to duplicate
informa􀆟on. We handled missing values in the dataset. We dropped rows with missing
values since they could poten􀆟ally distort our analysis and predic􀆟ons. We applied Outlier
Detec􀆟on and Removal Using Z-Score method. The Z-score is a measure of how many
standard devia􀆟ons an element has from the mean. It is a common method to detect and
remove outliers in a dataset. In our data set every data point that has a z-score greater than
or less than a threshold -4 and +4 can be considered an outlier. In our case we did not
remove outliers because they were few and we judged them to be important for the
performance of the models.

## Methodologies Applied

### Random Forest Regression Analysis
We performed a random forest regression analysis based on selected features to
provide informa􀆟on about the importance of each feature in CO2 emissions. These
importances represent the rela􀆟ve contribu􀆟on of each feature to the predic􀆟on
task as determined by the trained Random Forest Regressor model. It indicates that
"Energy produc􀆟on" 41.5% and "Energy consump􀆟on" 56.1% are the most
important features for predic􀆟ng CO2 emissions, while "Popula􀆟on" and "GDP" have
compara􀆟vely less influence.

2) We use a data mining workflow for classifica􀆟on using a decision tree and random
forest classifier. It filters the dataset, performs data preprocessing, trains the
classifier, predicts CO2 emission categories, evaluates the model's performance, and
provides visualiza􀆟ons of feature importances and confusion matrix.

### Classification: Decision Tree & Random Forest
We use a data mining workflow for classifica􀆟on using a decision tree and random
forest classifier. It filters the dataset, performs data preprocessing, trains the
classifier, predicts CO2 emission categories, evaluates the model's performance, and
provides visualiza􀆟ons of feature importances and confusion matrix.

Decision Tree Classifier:
Accuracy: The accuracy of the model is 0.9551, indica􀆟ng that it correctly predicts
the CO2 emission category for approximately 95.51% of the instances in the test set.
Classifica􀆟on Report: The classifica􀆟on report provides a detailed assessment of the
model's performance for each CO2 emission category.
Metrics: For the 'high' category, the precision, recall, and F1-score are all around
0.98, indica􀆟ng high performance.
For the 'low' category, the precision is 0.90, the recall is 0.96, and the F1-score is
0.93. These metrics suggest that the model performs well but with slightly lower
precision compared to the 'high' category.
For the 'medium' category, the precision is 0.77, the recall is 0.67, and the F1-score
is 0.72. These metrics suggest that the model has rela􀆟vely lower performance for
the 'medium' category.
The macro avg F1-score is 0.88, indica􀆟ng overall good performance. The weighted
avg F1-score is 0.95, which takes into account class imbalance and provides an
average F1-score weighted by support.

Random Forest Classifier:
Accuracy: The accuracy of the model is 0.97, indica􀆟ng that it correctly predicts the
CO2 emission category for approximately 97% of the instances in the test set.
Classifica􀆟on Report: The classifica􀆟on report shows high precision, recall, and F1-
score values for all categories. The metrics are consistently above 0.94 for precision,
recall, and F1-score for each category, indica􀆟ng strong performance.
The macro avg F1-score is 0.97, indica􀆟ng excellent overall performance. The
weighted avg F1-score is 0.97, taking into account class imbalance, and providing an
average F1-score weighted by support.
In summary, both models perform well, but the Random Forest Classifier
demonstrates slightly be􀆩er overall performance compared to the Decision Tree
Classifier, as reflected by higher accuracy and F1-scores.

### Association Rule Mining: Apriori Algorithm
We apply a data mining workflow for associa􀆟on rule mining using the Apriori
algorithm. This model performs associa􀆟on rule mining on a dataset with categorical
variables. It filters the dataset, discre􀆟zes the CO2 emission column, encodes
transac􀆟ons, finds frequent itemsets using the Apriori algorithm, generates
associa􀆟on rules, filters the rules based on desired rela􀆟onships, and prints the
filtered rules.

In summary, the results of associa􀆟on rule mining using the Apriori algorithm
provide insights into the rela􀆟onships between different energy types and the CO2
emission category of "high." Here's a brief summary of the findings:
Energy type= petroleum_n_other_liquids -> CO2_emission=high:
The presence of the energy type "petroleum_n_other_liquids" in a transac􀆟on
indicates a high likelihood (confidence of 78.33%) of also having a "high" CO2
emission category.

This associa􀆟on has a strong posi􀆟ve rela􀆟onship (li􀅌 of 1.93), meaning that the
occurrence of "petroleum_n_other_liquids" is significantly related to a "high" CO2
emission category.

Energy type=natural_gas -> CO2_emission=high:
The presence of the energy type "natural_gas" in a transac􀆟on suggests a moderate
likelihood (confidence of 46.07%) of having a "high" CO2 emission category.
This associa􀆟on has a posi􀆟ve rela􀆟onship (li􀅌 of 1.14), indica􀆟ng that "natural_gas"
and a "high" CO2 emission category are somewhat related.
These findings suggest that there are notable associa􀆟ons between specific energy
types and the CO2 emission category of "high." The presence of
"petroleum_n_other_liquids" demonstrates a strong posi􀆟ve rela􀆟onship with a
"high" CO2 emission category, while the presence of "natural_gas" indicates a
moderate associa􀆟on. These associa􀆟ons can provide insights into the impact of
different energy types on CO2 emissions and can inform decision-making in energy
and environmental policies.

### Clustering: KMeans Algorithm
Clustering analysis to group con􀆟nents based on their total CO2 emissions. This
model clusters con􀆟nents based on their total CO2 emissions using the KMeans
algorithm. It groups the data by con􀆟nent and calculates the sum of CO2 emissions
for each con􀆟nent. Then, it applies KMeans clustering to assign con􀆟nents to one of
the three clusters based on their CO2 emission levels. Based on the results, the
con􀆟nents are grouped into three clusters:
Cluster 0: This cluster consists of Africa, Oceania, and South America. These
con􀆟nents have rela􀆟vely lower total CO2 emissions compared to the other clusters.
Cluster 1: This cluster includes Europe and North America. These con􀆟nents have
higher total CO2 emissions compared to Cluster 0 but lower compared to Cluster 2.
Cluster 2: This cluster consists of Asia. Asia has the highest total CO2 emissions
among all con􀆟nents. This informa􀆟on can be useful for understanding global
emissions trends, iden􀆟fying regions with similar emission profiles, and informing
policy decisions related to environmental sustainability and climate change.

### Regression Analysis: XGBoost
We perform regression analysis using the XGBoost algorithm to predict CO2
emissions based on energy consump􀆟on, energy produc􀆟on, GDP, and popula􀆟on
data. This model trains an XGBoost regression model to predict CO2 emissions and
evaluates its performance using mean squared error. It then uses the trained model
to predict CO2 emissions for the year 2020 for each con􀆟nent and the top 11
countries with the highest CO2 emissions. For this model we did not standardize the
data because tree-based models like XGBoost are o􀅌en considered scale invariant,
meaning that they can handle features on different scales. This is because these
models are based on a series of binary splits, so the actual scale of the features does
not necessarily impact the model's performance.

### Time Series Forecasting: ARIMA
Utilized the ARIMA model to foresee CO2 emissions for each continent in the years 2020 through 2023, assisting in understanding future emission trajectories and potential implications.

## Results & Insights
The application of various data mining and predictive analytics methodologies brought forth a multitude of insights and predictions regarding global energy consumption and CO2 emissions. The detailed results of each methodology, along with visualizations, are discussed in the respective Jupyter notebooks.

## Conclusions
The use of XGBoost for predic􀆟on in this coursework was the most appropriate because the
dataset contains con􀆟nuous values. Given that the target variable in this case is con􀆟nuous
(CO2 emissions), using XGBoost as a regression model is a suitable choice. XGBoost is
designed to handle regression problems by op􀆟mizing the model to minimize the difference
between predicted and actual con􀆟nuous values. It provides flexibility, strong predic􀆟ve
performance, and the ability to capture complex rela􀆟onships in the data, making it wellsuited
for this predic􀆟on task. To op􀆟mize our parameters we used GridSearchCV from
scikit-learn, where the XGBoost model is passed as the es􀆟mator, the parameter grid is
specified, and cross-valida􀆟on is applied with 5 folds. The scoring metric is set to nega􀆟ve
mean squared error (neg_mean_squared_error) to op􀆟mize for lower values.
Best Parameters:
{'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 500}
Improving MSE:197096.75906577002 to MSE: 156742.2987018007
Applying best parameters to our xgboost model we had much be􀆩er performance reducing
MSE and improving our predic􀆟ons.
The most challenge part of this coursework was to select a related data set and understand
it. Furthermore, the Data Preprocessing procedure was really confusing trying to use the
best method to fill all the NaN values. Firstly we applied interpolate method:
df = data.groupby('Country').apply(lambda group: group.interpolate(method='linear',
limit_direc􀆟on='both')
The interpolate() method uses various interpola􀆟on techniques to fill the missing values. The
default interpola􀆟on method is linear interpola􀆟on, which fills the missing values with a
linearly interpolated value based on the neighboring data points. However, in the
con􀆟nua􀆟on of the coursework we found that the results were not as correct as using
dropna() method. We realized that the most important factor in properly analyzing data and
applying machine learning methods effec􀆟vely is having correct and clean data. Finally,
standardizing our data helped us to improve model performance, make algorithm
convergence quicker and make the model more interpretable.

## Setup & Requirements
Instructions on how to clone the repository, install necessary packages (Pandas, NumPy, Matplotlib, Plotly, Scikit-learn, XGBoost, Statsmodels, etc.), and execute notebooks/scripts.


