# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about banking customers including their age, gender, job, marital status and also marcroeconomic variables such as consumer price index and consumer confidence index. In this project, we seel to predict whether or not an individual will make a term deposit at the bank.

The best performing model was identified through AutoMl and it was a prefitted soft voting ensemble classfier with an accuracy of 0.9176. Therefore, this model would predict whether an individual would make a term deposit with an accuracy of 92%.

![image](https://user-images.githubusercontent.com/38438203/118903863-a0852000-b8e6-11eb-8ece-c843867f02a9.png)

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

In this pipeline, we first create the train.py script which is used to do the following:
  1. Import the dataset using the TabularDatasetFactory class 
  2. Clean and transform it using the clean_data function
  3. Split it into training and test sets using scikit-learn's train_test_splitfunction and 
  4. Define an accuracy maximizing scikit-learn logistic regression model which accepts two user inputted hyperparameters - the inverse reguarization strength            ('C') and the naximum number of iterations ('max_iter')
  
  We then open a Jupyter Notebook called udacity-project.ipynb on the Azure ML platform and set it to run on a CPU powered 'STANDARD_DS3_v2' compute instance. We     then define our worskspace from the default configuration using the from_config method and also define our experiment calling it banking-classification. After we   define our workspace, we use the Azure ML SDK to spin up a 'STANDARD_D2_V2' CPU powered compute cluster with a maximum of 4 nodes. After creating the compute       cluster, we define the configuration for our hyperdrive using the HyperDriveConfig method to which we pass our train.py script, a random parameter sampler for the   'C' and 'max_iter' hyperparameters with a bandit stopping policy all for the aim of maximizing accuracy for our classification task. After defining the             configuration for the hyperdrive, we submit our experiment with this configuration and let it run. After the experiments finishes running, we identify the best     performing model and save it.

**What are the benefits of the parameter sampler you chose?**
  The reason we used a random parameter sampler is beacuse it can help identify the best hyperparameters in shorter time than an exhastive grid search. Random         sampling also searches more of the hyperparameter space that a grid search if the grid search is poorly defined.

  Bandit policy is an early termination policy based on slack factor and evaluation interval. Bandit ends runs when the primary metric isn't within the specified   slack factor of the most succesful run.

## AutoML
To use AutoML, we read the dataset directly into the Jupyter Notebook and pass it into the clean_data function of train.py. We then concatenate the predictor variables and the labels into pandas dataframe based training set. After creating this training set, we define the AutoML configuration using the AutoMLConfig method by passing in the training set identifying labels column, setting the task as classification, the primary metric which we need to optimize as accuracy, and the number of cross validations to be done as three. We then submit the AutoML experiment and let it run. After it finishes running, we identify the best model and save it.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
In comparing the hyperdrive pipeline results with the AutoML results, although both pipelines provided model results with high accuracies, we noticed that the best performing AutoML model with was a prefitted soft voting ensemble classfier had a slightly higher accuracy than the hyperdrive best model with C = 0.5 and max_iter = 50. The AutoML model had an accuracy of 0.92 versus the hyperdrive model with an accuracy of 0.91. The reason AutoML resulted in slightly better accuracy was because it ran through 50 different calssification models some of which were ensemble models whereas the hyperdrive model was a simple logistic regression mode. It is important to note that the AutoML pipeline took much longer to run than the hyperdrive pipeline. A question worth considering is whether it was worth running AutoML if the faster hyperdrive provided good enough accuracy. This might be more of an issue for larger datasets.


## Future work
We can try running the experiments again using different features, optimizing a different metric instead of accuracy such as maybe the F1 score, run hyperdrive using ensemble models such as XGBoost or RandomForest, or maybe even use deep learning in the AutoML.

## Proof of cluster clean up
The cluster was deleted at the end using the delete method.
