

# Installation instructions

  

After cloning the repository, Make sure you have Python3.

 

-  [https://www.python.org/downloads/](https://www.python.org/downloads/)



Create a new virtual environment and activate it.


  

    python3 -m venv /path/to/new/virtual/environment_name
    source /path/to/new/virtual/environment_name/bin activate

Now install all the required libraries. Make sure you're inside the project directory before running the command.

  

    pip install -r requirements.txt

  
  
  # Running instructions
  
Run program using python or python3

    python main.py

  # Codebase and outputs


 1. Codebase - contains the entire codebase with cgm data, helper functions and config
 2. Outputs - contains .csv files of feature matrices, PCA feature matrices, PCA eigenvectors and text file for PCA eigenvalues
 3. Plots - contains folders for the plots for each feature and PCA features
 4. README.pdf - Quick start readme file.
