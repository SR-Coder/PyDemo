import sys, scipy, numpy, matplotlib, pandas, sklearn

# This is a machine learning mastery.com lesson.
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# huge props to Jason Browniee
# Check Versions
print('Python: {}'.format(sys.version))
print('SciPy: {}'.format(scipy.__version__))
print('Numpy: {}'.format(numpy.__version__))
print("Matplotlib: {}".format(matplotlib.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Sklearn: {}'.format(sklearn.__version__))

# Import libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load a dataset directly from the UCI Machine Learning Repo
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# now to look at the dataset 
#   1. Dimension of the dataset
#   2. Peel at the data itself
#   3. satistical summary of all attributes
#   4. Breakdown of the data by the class variable

# shape
print(dataset.shape)

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())

# box and wisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

# histogram plots
dataset.hist()

# scatter plot matrix
scatter_matrix(dataset)

pyplot.show()