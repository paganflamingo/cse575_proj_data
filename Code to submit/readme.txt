Step 1: Run the Preprocessing.py file. This generates an 'output.csv' file which will have the processed data.
Make sure the data file, 'cumulative_2020.09.04_15.26.41.csv' and the Preprocessing.py file are in the same folder.

Step 2: Run the corresponding file to run the classification algorithm on the 'output.csv'data.

To run Random Forest Classifier: Run RandomForest_withCV.py file. Make sure that the 'output.csv' and 'RandomForest_withCV.py' files are in the same folder.

To run Naive Baye's Classifier:
  1) download python and package prerequisites
  2) 'python3 NaiveBayesResults.py' should result in the resulting f1 scores, accuracy, and confusion matrix scores
  3) Make sure that the 'output.csv' file is in the same folder/the directory for loading in data is updated to the proper location with 'output.csv'

To run AdaBoost Classifier:
  1) download python and package prerequisites
  2) 'python 3 adaboost.py' should result in the confusion matrices and accuracy results. 
  3) Make sure that the 'output.csv' file is in the same folder/the directory for loading in data is updated to the proper location with 'output.csv'

To run Logistic Regression Classifier:

To run Voting Classifier:

To run Neural Network Classifier:
