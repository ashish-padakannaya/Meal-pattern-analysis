'''
Python file to create plots for each of the features generated and plots for PCA
'''
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
import os

# patient_features_df = pd.read_csv('Outputs/patient_features.csv')

def plot_explained_variance_pca():
    Y = []
    with open('Outputs/explained_variance_ratio.txt', 'r') as f:
        Y = [float(i) for i in f.read().splitlines()] 
    X = ['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5']
    plt.bar(X,Y)
    plt.ylim(0,1)
    plt.title("Explained variance of each principal component")
    plt.savefig("Plots/PCA_plots/pca_explained_variance.png", dpi=200)
    plt.close()

def plot_eigenvectors_pca():
    df = pd.read_csv("Outputs/pca_components.csv")
    del df['Unnamed: 0']
    pcs = ['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5']
    vals = []
    for index, rows in df.iterrows():
        vals += list(rows.values)
    barlist = plt.bar(list(range(len(vals))), vals)
    color_map = {
        0 : '#853326',
        1 : '#3d65a6',
        2 : '#164f33',
        3 : '#abab02',
        4 : '#bc82e8',
    }
    print(len(vals))
    for i in range(len(vals)):
        barlist[i].set_color(color_map[int(i/7)])

    red_patch = mpatches.Patch(color='#853326', label='PC-1')
    blue_patch = mpatches.Patch(color='#3d65a6', label='PC-2')    
    green_patch = mpatches.Patch(color='#164f33', label='PC-3')    
    yellow_patch = mpatches.Patch(color='#abab02', label='PC-4')    
    purple_patch = mpatches.Patch(color='#bc82e8', label='PC-5')
    plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch, purple_patch])

    plt.title("Eigenvectors for 5 pca features")
    plt.savefig("Plots/PCA_plots/pca_eigenvectors" + ".png", dpi=400)
    plt.close()


def plot_features(df):
    
    features = list(df.columns)[2:]
    for pat_num in [1,2,3,4,5]:
        patient = df[(df.patient_number == pat_num)]
        for dimension in features:
            meals_range = patient.meal_number.to_numpy()
            values = patient[dimension].to_numpy()
            
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.scatter(meals_range, values, label=dimension, marker='o')
            ax.plot(meals_range, values)
            ax.set_title( str.format('{0} feature for Patient {1}', dimension, str(pat_num)))
            plt.xlabel('Meal number')
            plt.ylabel('Feature values')
            plt.legend()
            if not os.path.exists(os.getcwd() + '/Plots/' + dimension + '_plots'):
                    os.makedirs(os.getcwd() + '/Plots/' + dimension + '_plots')
            plt.savefig('Plots/' + dimension + '_plots' + '/patient' + str(pat_num) + '_' + dimension +'.png', dpi=200)
            plt.close()


def makePlots():
    df = pd.read_csv('Outputs/patient_features.csv')
    plot_explained_variance_pca()
    plot_eigenvectors_pca()
    plot_features(df)

