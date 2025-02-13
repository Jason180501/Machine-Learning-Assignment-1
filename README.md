# Supervised-Machine-Learning-Analysis

In this project, we analyze supervised machine learning approaches using the `Perinatal Risk Information-1.csv` dataset.


## Understanding the Dataset:

* The given dataset (`Perinatal Risk Information.csv`) includes health parameters which help us judge the perinatal health risk level based on the Age, Systolic BP, Diastolic BP, BS, BodyTemp and HeartRate. This risk level classification is stored in the Type column. The problem statement asks us to correctly classify whether a woman’s maternal health is at high risk, mid risk or low risk.
* The independent variables in this dataset are the columns labeled Age, Systolic BP, Diastolic BP, BS, BodyTemp and HeartRate.
* The dependent variable in this dataset is the column Type.
* Classification would be the appropriate machine learning approach for this problem as we have a dependent variable Type which takes 3 inputs - high risk, mid risk and low risk. Training a classifying model using supervised learning would be the optimal solution for this dataset as we wish to build a model which can accurately determine these factors.

## Data Exploration and Pre-Processing

* There are a total of 7 columns and 1014 entries (rows) in the dataset.
* From the observations done on the Type column of the dataset we find that the dataset is imbalanced. Low risk has 406 counts, mid risk has 336 counts, and high risk has 272 counts. This shows that the dataset has quite an imbalance in it as in a balanced dataset the counts would be relatively equal between all the classes.
* When checking if there were any NULL values present in the dataset using the is.null() command we found that there were no NULL values present in the dataset.
* When looking at the data using the dataset.describe() function we find that for the column HeartRate there is a minimum value of 7.0, which is extremely low and implausible and could be an outlier. So, to amend this we set any values less than 40 in this column to the median of the column.
* These data pre-processing steps are required so as to get a good dataset without any outliers which could induce bias to our model.
* While overviewing the other columns there are no obvious outliers present in the dataset, so we proceed with splitting the dataset.
* We split the dataset using the train_test_split() function allotting 80% of the dataset to the training set and 20% of the dataset to the test set. This gives us our training and test sets – X_train, y_train, X_test and y_test.
* X_train and X_test contain the data regarding the independent variables and y_train and y_test contain the labels of the dependent variable “Type”.
* X_train and y_train have 811 samples whereas X_test and y_test have 203 samples.


## Algorithm Selection and Application
* The classification algorithms being used to generate the classifiers are the `Decision Tree Classifier` and `Random Forest Classifier`.
* Decision Tree Classifier – A decision tree is a non-parametric supervised learning algorithm, which is utilized for both classification and regression tasks. It has hierarchical tree structure, which consists of a root node, branches, internal nodes and leaf nodes. (IBM - https://www.ibm.com/topics/decision-trees)
* Random Forest Classifier – Random Forest is a machine learning algorithm, that combines the output of multiple decision trees to reach a single result. It handles both classification and regression problems. (IBM - https://www.ibm.com/topics/random-forest)
* In the model parameter we set class_weight = “balanced” to try to remove any imbalance in the dataset.
* We apply hyperparameter tuning for the decision tree classifier. We apply the tuning using the RandomizedSearchCV library. On evaluating the Decision tree classifier with the hyperparameter tuning, we get an accuracy of around 78%. We also observe a precision of 77%, recall of 81% and f1-score of 78%.
When the max_depth was kept at less than 100 we saw a loss in accuracy to around 75%. But when the max_depth was kept more than 100 we still saw a loss in accuracy to around 74%. Similarly, when declaring the class_weight as balanced we get the current accuracy of 77% but when it is not balanced there is a drop in accuracy to around 75%.

## Model Evaluation and Comparative Analysis
* On evaluating the Decision tree classifier, we get an accuracy score of around 81%. We also observe a precision of 80%, recall of 83% and f1-score of 81%.
* On evaluating the Random Forest Classifier, we get an accuracy score of around 80%. We also observe a precision of 79%, recall of 82% and f1-score of 80%.
* The model’s used provide similar results because the dataset is small, and the data is spread out well thus preventing overfitting in the two algorithms.

## Ethical Considerations
* Machine Learning models depend on the data they are trained on. If bias is present in the data, these biases can be amplified in the model’s predictions. For example, In the case of maternal health risks, if the model is trained on data of women below the age of 30, the model might underperform for women over the age of 30.
* Machine Learning models must be approached with caution because even though there could be many benefits like early detection of risks, bias could always play a part in misclassification. These risks are not to be taken lightly and could even lead to misdiagnosis. These ethical considerations need to be addressed to ensure that the machine learning tools are used responsibly and in the right manner.


