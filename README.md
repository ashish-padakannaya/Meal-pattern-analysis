  

# Installation instructions

  

After cloning the repository, Make sure you have Python3 and pip install.

 

-  [https://www.python.org/downloads/](https://www.python.org/downloads/)
 - https://pip.pypa.io/en/stable/installing/

Now install all the required libraries. Make sure you're inside the project directory before running the command.

  

    pip install -r requirements.txt


  # Pre - running instructions
  Keep the input data, which is a CSV file, **in the same directory as `main.py`**
  
  # Running instructions
  
Run program from inside the project directory using python or python3

    python main.py <filename>

**For eg.**

    python main.py testsampleinput.csv

filename is the name of the CSV you will put as input for testing data.

  # Codebase and outputs


 1. Codebase - contains the entire codebase with 4 classifier models and meal/no-meal data.
 2. Outputs - classifier labels are stored in **classifier_predictions.csv** for each classifier.
 
	 Meal label is signified by 1 and No Meal is signified by 0.
 
 4. README.pdf - Quick start readme file.

<br/>

# Average scores for different classifiers with K-Fold Validation:


|	|F1 Score	|Accuracy|Recall	|Precision 
|--|--|--|--|--|
|Decision Tree| 69.15 | 67.15 |70.19|70.38|
|Multilayer Perceptron| 73.14 |70.26  |72.43|75.36|
|Random Forest|69.5|69.27|67.05|71.86|
|Support Vector Machine| 65.59 |66.33|63.98|67.12|
