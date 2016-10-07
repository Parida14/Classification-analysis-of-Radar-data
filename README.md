TAMU ISEN 613: Engineering Data Analysis final project

PROJECT STATEMENT
To differentiate between the radar returns from the ionosphere into good return and bad returns.  Based on definition, “"Good" radar returns are those showing evidence of some type of structure in the ionosphere, while, "Bad" returns are those that do not and their signals pass through the ionosphere”. This is a classification problem, thus all techniques used in this study will use classification methods. 

DATASET
Our data set consists of 351 cases each having 35 attributes. The data set consists of integer and real values attributes there is no missing values for any of the attributes. This data is available at UCI Machine Learning Repository under name “Ionosphere Data Set”

APPROACH
Firstly, after loading the data set we cleaned the data set by checking for missing values and removing the entries with missing values from the data set, for more accurate analysis. Then we reduced the number of variables using several dimension reduction techniques such as Principle Component Analysis (PCA), Support Vector Machine Ranking (SVM Ranking) and RandomForest.
We employed a new package “FSelector” and also a new technique – “Partial Least Square (PLS)”, in our analysis to check for important features. Once the important variables are selected, we split the data set into Training and Test data sets, with the training data set having random 80% data points of the actual data and the test data having the 20%. 
As the next step, Classification will be done on the training data set using techniques like Logistic Regression, Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA) and K-Nearest Neighbors (KNN). Multiple iterations of each of the above mentioned techniques may be used to reach the best accurate models. All the while finding out the misclassification rate each time (Cross Validation) and the least misclassification rate will determine the best model.
