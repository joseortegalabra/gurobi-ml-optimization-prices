## Examples Gurobi ML

This repo contains codes to understand how to use gurobi pandas and gurobi ml to solve optimization problems where one or more constraints are defined as machine learning models (a conection between machine learning and operational research)

### Principal content
This repo contains codes to optimize the income of selling a product in different regions considering the supply, demand and costs. 

To optimize the income is defined as a optimization problem which objetive is optimize the income suject to demand constraint, supply contraints, etc.

Also, a Machine learning model is used to predict the demand and this model is one constraint more in the optimization problem. To do this a package gurobi-machine-learning is used


### Principal Examples
In the folder "optimization" there 3 differents examples (to run this examples it is necesary trainning the machine learning models previosly, this is done in the folder "model_ml")
- **one_ml_model**: in this notebook is created a optimization problem where one machine learning model is used to predict the demand in all regions
- **multiple_ml_models**: in this notebook is created a optimization problem where multiple machine learning models is used to predict the demand in all regions, one model for region. Also, the model see the prices of all regions to predict the demand of one region
- **multiple_ml_models_time_period**: is used the same previous notebook but it is modified to optimize across a time horizon


### Folders
In this repo, there are the following folders:

- **0_oficial_examples**: this folder contains oficial notebooks of examples codes of gurobi machine learning and other notebooks to explore this codes

- **1_data**: it contains a notebook to generate a data used in the following folders to generate machine learning models and optimization models

- **2_eda**: exploratory data analysis of the data generated in the previous notebook

- **3_feature_eng**: feature engineering of the data generated, such as, deleting some rows and columns of the data. Also in this folder is generated a second dataset that will be use to train ml models. So, there will generated two datasets: the firts one "basic_features" that contains the rows price, demand, region; the second datasets "price_regions" that contains the rows demand, region and prices for each region. The firts dataset is used to predict the demand

- **4_model_ml**: This folders contains differents notebooks to train different machine learning models with differents datasets
    - **1_basic_features_one_lr**: one linear regression is trained with dataset "basic_features"
    - **2_basic_features_multiple_lr**: multiple linear regressions are trained, one for each region
    - **3_basic_features_one_gbr**: it remplaces the one linear regression with a gradient boosting model
    - **4_prices_regions_one_lr**: this notebooks use the dataset "price_regions" that containt the prices of each regions to predict the demand of one region. It follows the same logic, in this case a linear regression is used for all regions
    - **5_prices_regions_multiple_lr**: it uses multiple linear regressions to predict the demand for each region, one model for region
    - **6_prices_regions_one_grb**: it replace the regresion in notebook 4 for a gradient boosting model

- **5_optimization**: it contains the notebook of optimization model that use the machine learning models as constraints. The optimizations was explained before in section "Principal Examples"
    - **one_ml_model**: this optimization use the model trainied in "1_basic_features_one_lr"
    - **multiple_ml_models**: this optimization use the models trained in "5_prices_regions_multiple_lr"
    - **multiple_ml_models_time_period**: this optimization also use the models trained in "5_prices_regions_multiple_lr"

- **6_streamlit**: this folder contains the same code of optimization develop in folder "5_optimization/multiple_ml_models" and its showed in a simple streamlit app

- **artifacts**: in this folder there are artifacts (data and models) used in the previous notebooks. This artifacts are not uploaded to the repo but its can be generated running the notebook 