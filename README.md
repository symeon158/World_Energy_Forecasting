# Energy Forecasting: Data Mining & Predictive Analysis

## Project Overview
This repository explores various data mining and machine learning approaches towards predicting, classifying, and understanding the relationships and patterns in global energy consumption and CO2 emissions. Leveraging a rich dataset, we delve deep into Random Forest Regression, Decision Trees, Clustering, Association Rule Mining, and Time Series Forecasting to unearth insights and forecasts pertinent to World Energy dynamics. *If you want to see all the plots download the html file. [Download HTML file](https://github.com/symeon158/World_Energy_Forecasting/raw/main/Data%20Mining%20Symeon_Papadopoulos.html)


## Table of Contents
1. [Introduction](#introduction)
2. [Data Loading](#data-loading)
3. [Data Preprocessing](#data-preprocessing)
4. [Methodologies Applied](#methodologies-applied)
   - [Random Forest Regression Analysis](#random-forest-regression-analysis)
   - [Classification: Decision Tree & Random Forest](#classification-decision-tree--random-forest)
   - [Association Rule Mining: Apriori Algorithm](#association-rule-mining-apriori-algorithm)
   - [Clustering: KMeans Algorithm](#clustering-kmeans-algorithm)
   - [Regression Analysis: XGBoost](#regression-analysis-xgboost)
   - [Time Series Forecasting: ARIMA](#time-series-forecasting-arima)
5. [Results & Insights](#results--insights)
6. [Conclusions](#conclusions)
7. [Setup & Requirements](#setup--requirements)

## Introduction
The global challenge of reducing CO2 emissions and adapting to a sustainable energy paradigm makes it imperative to understand the patterns, associations, and forecasting of energy consumption and emissions. This project employs various data mining techniques to dissect and predict energy-related variables, with a particular focus on CO2 emissions across different continents.

## Data Loading
We began by loading our dataset, which contains information about CO2 emissions, energy consumption, production, GDP, population, energy intensity per capita and energy intensity by GDP for different countries from 1980 to 2019.

## Data Preprocessing
First, we checked the data for duplicates to avoid biasing the performance of the analysis and machine learning model by giving more weight to duplicate information. We handled missing values in the dataset. We dropped rows with missing values since they could potentially distort our analysis and predictions. We applied Outlier Detection and Removal Using Z-Score method. The Z-score is a measure of how many standard deviations an element is from the mean. It's a common method to detect and remove outliers in a dataset. In our data set every data point that has a z-score greater than or less than a threshold -4 and +4 can be considered an outlier. In our case we did not remove outliers because they were few and we judged them to be important for the performance of the models.

## Methodologies Applied

### Random Forest Regression Analysis
We performed a random forest regression analysis based on selected features to provide information about the importance of each feature in CO2 emissions. These importances represent the relative contribution of each feature to the prediction task as determined by the trained Random Forest Regressor model. It indicates that "Energy production" 41,5% and "Energy consumption" 56.1% are the most important features for predicting CO2 emissions, while "Population" and "GDP" have comparatively less influence.

### Classification: Decision Tree & Random Forest
We use a data mining workflow for classification using a decision tree and random forest classifier. It filters the dataset, performs data preprocessing, trains the classifier, predicts CO2 emission categories, evaluates the model's performance, and provides visualizations of feature importances and confusion matrix.

For the Decision Tree Classifier:
Accuracy: The accuracy of the model is 0.9551, indicating that it correctly predicts the CO2 emission category for approximately 95.51% of the instances in the test set.
Classification Report: The classification report provides a detailed assessment of the model's performance for each CO2 emission category.
Metrics: For the 'high' category, the precision, recall, and F1-score are all around 0.98, indicating high performance.
For the 'low' category, the precision is 0.90, the recall is 0.96, and the F1-score is 0.93. These metrics suggest that the model performs well but with slightly lower precision compared to the 'high' category.
For the 'medium' category, the precision is 0.77, the recall is 0.67, and the F1-score is 0.72. These metrics suggest that the model has relatively lower performance for the 'medium' category.
The macro avg F1-score is 0.88, indicating overall good performance. The weighted avg F1-score is 0.95, which takes into account class imbalance and provides an average F1-score weighted by support.

For the Random Forest Classifier:
Accuracy: The accuracy of the model is 0.97, indicating that it correctly predicts the CO2 emission category for approximately 97% of the instances in the test set.
Classification Report: The classification report shows high precision, recall, and F1-score values for all categories. The metrics are consistently above 0.94 for precision, recall, and F1-score for each category, indicating strong performance.
The macro avg F1-score is 0.97, indicating excellent overall performance. The weighted avg F1-score is 0.97, taking into account class imbalance, and providing an average F1-score weighted by support.
In summary, both models perform well, but the Random Forest Classifier demonstrates slightly better overall performance compared to the Decision Tree Classifier, as reflected by higher accuracy and F1-scores.

### Association Rule Mining: Apriori Algorithm
We apply a data mining workflow for association rule mining using the Apriori algorithm. This model performs association rule mining on a dataset with categorical variables. It filters the dataset, discretizes the CO2 emission column, encodes transactions, finds frequent itemsets using the Apriori algorithm, generates association rules, filters the rules based on desired relationships, and prints the filtered rules.

In summary, the results of association rule mining using the Apriori algorithm provide insights into the relationships between different energy types and the CO2 emission category of "high." Here's a brief summary of the findings:

Energy type= petroleum_n_other_liquids -> CO2_emission=high:
The presence of the energy type "petroleum_n_other_liquids" in a transaction indicates a high likelihood (confidence of 78.33%) of also having a "high" CO2 emission category.
This association has a strong positive relationship (lift of 1.93), meaning that the occurrence of "petroleum_n_other_liquids" is significantly related to a "high" CO2 emission category.

Energy type=natural_gas -> CO2_emission=high:
The presence of the energy type "natural_gas" in a transaction suggests a moderate likelihood (confidence of 46.07%) of having a "high" CO2 emission category.
This association has a positive relationship (lift of 1.14), indicating that "natural_gas" and a "high" CO2 emission category are somewhat related.

These findings suggest that there are notable associations between specific energy types and the CO2 emission category of "high." The presence of "petroleum_n_other_liquids" demonstrates a strong positive relationship with a "high" CO2 emission category, while the presence of "natural_gas" indicates a moderate association. These associations can provide insights into the impact of different energy types on CO2 emissions and can inform decision-making in energy and environmental policies.

### Clustering: KMeans Algorithm
Clustering analysis to group continents based on their total CO2 emissions. This model clusters continents based on their total CO2 emissions using the KMeans algorithm. It groups the data by continent and calculates the sum of CO2 emissions for each continent. Then, it applies KMeans clustering to assign continents to one of the three clusters based on their CO2 emission levels. Based on the results, the continents are grouped into three clusters:

Cluster 0: This cluster consists of Africa, Oceania, and South America. These continents have relatively lower total CO2 emissions compared to the other clusters.

Cluster 1: This cluster includes Europe and North America. These continents have higher total CO2 emissions compared to Cluster 0 but lower compared to Cluster 2.

Cluster 2: This cluster consists of Asia. Asia has the highest total CO2 emissions among all continents. This information can be useful for understanding global emissions trends, identifying regions with similar emission profiles, and informing policy decisions related to environmental sustainability and climate change.

### Regression Analysis: XGBoost
We perform regression analysis using the XGBoost algorithm to predict CO2 emissions based on energy consumption, energy production, GDP, and population data. This model trains an XGBoost regression model to predict CO2 emissions and evaluates its performance using mean squared error. It then uses the trained model to predict CO2 emissions for the year 2020 for each continent and the top 11 countries with the highest CO2 emissions. For this model we did not standardize the data because tree-based models like XGBoost are often considered scale invariant, meaning that they can handle features on different scales. This is because these models are based on a series of binary splits, so the actual scale of the features doesn't necessarily impact the model's performance.

### Time Series Forecasting: ARIMA
Utilized the ARIMA model to foresee CO2 emissions for each continent in the years 2020 through 2023, assisting in understanding future emission trajectories and potential implications.

## Results & Insights
The application of various data mining and predictive analytics methodologies brought forth a multitude of insights and predictions regarding global energy consumption and CO2 emissions. The detailed results of each methodology, along with visualizations, are discussed in the respective Jupyter notebooks and the pdf file. *If you want to see all the plots download the html file.

## Conclusions
The use of XGBoost for prediction in this coursework was the most appropriate because the dataset contains continuous values. Given that the target variable in this case is continuous (CO2 emissions), using XGBoost as a regression model is a suitable choice. XGBoost is designed to handle regression problems by optimizing the model to minimize the difference between predicted and actual continuous values. It provides flexibility, strong predictive performance, and the ability to capture complex relationships in the data, making it well-suited for this prediction task. To optimize our parameters we used GridSearchCV from scikit-learn, where the XGBoost model is passed as the estimator, the parameter grid is specified, and cross-validation is applied with 5 folds. The scoring metric is set to negative mean squared error (neg_mean_squared_error) to optimize for lower values.
Best Parameters:
{'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 500}


Improving MSE:197096.75906577002 to MSE: 156742.2987018007

Applying best parameters to our xgboost model we had much better performance reducing MSE and improving our predictions.
The most challenge part of this coursework was to select a related data set and understand it. Furthermore, the Data Preprocessing procedure was really confusing trying to use the best method to fill all the NaN values. Firstly we applied interpolate method:
df = data.groupby('Country').apply(lambda group: group.interpolate(method='linear', limit_direction='both')
The interpolate() method uses various interpolation techniques to fill the missing values. The default interpolation method is linear interpolation, which fills the missing values with a linearly interpolated value based on the neighboring data points. However, in the continuation of the coursework we found that the results were not as correct as using dropna() method. We realized that the most important factor in properly analyzing data and applying machine learning methods effectively is having correct and clean data. Finally, standardizing our data helped us to improve model performance, make algorithm convergence quicker and make the model more interpretable. 

## Setup & Requirements
Instructions on how to clone the repository, install necessary packages:
git clone [repository-link]

pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
plotly>=5.0.0
seaborn>=0.11.0
statsmodels>=0.12.0
xgboost>=1.4.0
scikit-learn>=0.24.0
mlxtend>=0.18.0



