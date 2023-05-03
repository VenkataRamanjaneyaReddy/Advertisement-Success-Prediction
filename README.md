# Advertisement-Success-Prediction

Advertisement Success Prediction
Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
Reading the Data
df = pd.read_csv('D:/ML/Advertisement Success Prediction/Dataset/advertising.csv')
df
Daily Time Spent on Site	Age	Area Income	Daily Internet Usage	Ad Topic Line	City	Male	Country	Timestamp	Clicked on Ad
0	68.95	35	61833.90	256.09	Cloned 5thgeneration orchestration	Wrightburgh	0	Tunisia	2016-03-27 00:53:11	0
1	80.23	31	68441.85	193.77	Monitored national standardization	West Jodi	1	Nauru	2016-04-04 01:39:02	0
2	69.47	26	59785.94	236.50	Organic bottom-line service-desk	Davidton	0	San Marino	2016-03-13 20:35:42	0
3	74.15	29	54806.18	245.89	Triple-buffered reciprocal time-frame	West Terrifurt	1	Italy	2016-01-10 02:31:19	0
4	68.37	35	73889.99	225.58	Robust logistical utilization	South Manuel	0	Iceland	2016-06-03 03:36:18	0
...	...	...	...	...	...	...	...	...	...	...
995	72.97	30	71384.57	208.58	Fundamental modular algorithm	Duffystad	1	Lebanon	2016-02-11 21:49:00	1
996	51.30	45	67782.17	134.42	Grass-roots cohesive monitoring	New Darlene	1	Bosnia and Herzegovina	2016-04-22 02:07:01	1
997	51.63	51	42415.72	120.37	Expanded intangible solution	South Jessica	1	Mongolia	2016-02-01 17:24:57	1
998	55.55	19	41920.79	187.95	Proactive bandwidth-monitored policy	West Steven	0	Guatemala	2016-03-24 02:35:54	0
999	45.01	26	29875.80	178.35	Virtual 5thgeneration emulation	Ronniemouth	0	Brazil	2016-06-03 21:43:21	1
1000 rows × 10 columns

Understanding the Data
df.dtypes
Daily Time Spent on Site    float64
Age                           int64
Area Income                 float64
Daily Internet Usage        float64
Ad Topic Line                object
City                         object
Male                          int64
Country                      object
Timestamp                    object
Clicked on Ad                 int64
dtype: object
df.shape
(1000, 10)
df.size
10000
df.columns
Index(['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage', 'Ad Topic Line', 'City', 'Male', 'Country',
       'Timestamp', 'Clicked on Ad'],
      dtype='object')
 
df.max()
Daily Time Spent on Site                           91.43
Age                                                   61
Area Income                                      79484.8
Daily Internet Usage                              269.96
Ad Topic Line               Visionary reciprocal circuit
City                                          Zacharyton
Male                                                   1
Country                                         Zimbabwe
Timestamp                            2016-07-24 00:22:16
Clicked on Ad                                          1
dtype: object
df.min()
Daily Time Spent on Site                                 32.6
Age                                                        19
Area Income                                           13996.5
Daily Internet Usage                                   104.78
Ad Topic Line               Adaptive 24hour Graphic Interface
City                                                Adamsbury
Male                                                        0
Country                                           Afghanistan
Timestamp                                 2016-01-01 02:52:10
Clicked on Ad                                               0
dtype: object
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 10 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Daily Time Spent on Site  1000 non-null   float64
 1   Age                       1000 non-null   int64  
 2   Area Income               1000 non-null   float64
 3   Daily Internet Usage      1000 non-null   float64
 4   Ad Topic Line             1000 non-null   object 
 5   City                      1000 non-null   object 
 6   Male                      1000 non-null   int64  
 7   Country                   1000 non-null   object 
 8   Timestamp                 1000 non-null   object 
 9   Clicked on Ad             1000 non-null   int64  
dtypes: float64(3), int64(3), object(4)
memory usage: 78.2+ KB
df.describe()
Daily Time Spent on Site	Age	Area Income	Daily Internet Usage	Male	Clicked on Ad
count	1000.000000	1000.000000	1000.000000	1000.000000	1000.000000	1000.00000
mean	65.000200	36.009000	55000.000080	180.000100	0.481000	0.50000
std	15.853615	8.785562	13414.634022	43.902339	0.499889	0.50025
min	32.600000	19.000000	13996.500000	104.780000	0.000000	0.00000
25%	51.360000	29.000000	47031.802500	138.830000	0.000000	0.00000
50%	68.215000	35.000000	57012.300000	183.130000	0.000000	0.50000
75%	78.547500	42.000000	65470.635000	218.792500	1.000000	1.00000
max	91.430000	61.000000	79484.800000	269.960000	1.000000	1.00000
df.corr()
Daily Time Spent on Site	Age	Area Income	Daily Internet Usage	Male	Clicked on Ad
Daily Time Spent on Site	1.000000	-0.331513	0.310954	0.518658	-0.018951	-0.748117
Age	-0.331513	1.000000	-0.182605	-0.367209	-0.021044	0.492531
Area Income	0.310954	-0.182605	1.000000	0.337496	0.001322	-0.476255
Daily Internet Usage	0.518658	-0.367209	0.337496	1.000000	0.028012	-0.786539
Male	-0.018951	-0.021044	0.001322	0.028012	1.000000	-0.038027
Clicked on Ad	-0.748117	0.492531	-0.476255	-0.786539	-0.038027	1.000000
df.nunique()
Daily Time Spent on Site     900
Age                           43
Area Income                 1000
Daily Internet Usage         966
Ad Topic Line               1000
City                         969
Male                           2
Country                      237
Timestamp                   1000
Clicked on Ad                  2
dtype: int64
df.isnull().any()
Daily Time Spent on Site    False
Age                         False
Area Income                 False
Daily Internet Usage        False
Ad Topic Line               False
City                        False
Male                        False
Country                     False
Timestamp                   False
Clicked on Ad               False
dtype: bool
Exploratory Data Analysis / Visualization
import missingno as no
no.bar(df, color='lightgreen')
<AxesSubplot:>

sns.heatmap(df.isnull(), yticklabels='False', cmap='Oranges')
<AxesSubplot:>

df.Male.value_counts()
0    519
1    481
Name: Male, dtype: int64
plt.hist(df['Age'],bins = 30)
plt.xlabel('Age')
plt.ylabel('Values')
plt.show()

plt.figure(figsize=(10,5))
sns.stripplot(x=df['Clicked on Ad'], y=df['Age'], palette='magma_r')
plt.show()

sns.violinplot(x=df['Clicked on Ad'], y=df.Age, palette='rainbow')
<AxesSubplot:xlabel='Clicked on Ad', ylabel='Age'>

sns.scatterplot(df['Daily Time Spent on Site'], df['Daily Internet Usage'], hue=df['Clicked on Ad'], data=df, palette='RdPu')
C:\Users\Raman\AppData\Local\Programs\Python\Python39\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
<AxesSubplot:xlabel='Daily Time Spent on Site', ylabel='Daily Internet Usage'>

plt.figure(figsize=(14,10))
hm = sns.heatmap(df.corr(), annot=True, cmap="RdYlBu")
plt.show()

sns.boxplot(df['Clicked on Ad'], df['Daily Internet Usage'], data=df, palette='gnuplot_r')
C:\Users\Raman\AppData\Local\Programs\Python\Python39\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
<AxesSubplot:xlabel='Clicked on Ad', ylabel='Daily Internet Usage'>

sns.FacetGrid(df, hue='Clicked on Ad', palette='magma').map(sns.distplot, 'Area Income').add_legend()
C:\Users\Raman\AppData\Local\Programs\Python\Python39\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
  warnings.warn(msg, FutureWarning)
C:\Users\Raman\AppData\Local\Programs\Python\Python39\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
  warnings.warn(msg, FutureWarning)
<seaborn.axisgrid.FacetGrid at 0x1e3314f5610>

sns.pairplot(df,hue='Clicked on Ad',palette='gist_rainbow')
<seaborn.axisgrid.PairGrid at 0x1e32f229370>

Splitting the Data into Dependent and Indpendent variables
x = df[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = df['Clicked on Ad']
x.shape
(1000, 5)
y.shape
(1000,)
Training and Testing the Data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=5)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
Logistic Regression
lm = LogisticRegression(random_state=5)
lm.fit(xtrain, ytrain)
LogisticRegression(random_state=5)
Prediction
ypred_train = lm.predict(xtrain)
ypred_test = lm.predict(xtest)
Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(ytest, ypred_test)
cm
array([[119,   7],
       [ 16, 108]], dtype=int64)
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="OrRd" ,fmt='g')
<AxesSubplot:>

Accuracy
print("Accuracy of training data:", accuracy_score(ytrain, ypred_train)*100)
ac1 = accuracy_score(ytest, ypred_test)*100
print("Accuracy of testing data:", ac1)
Accuracy of training data: 89.33333333333333
Accuracy of testing data: 90.8
Support Vector Machine
svc = SVC(C=10)
svc.fit(xtrain, ytrain)
SVC(C=10)
Prediction
ypred_train = svc.predict(xtrain)
ypred_test = svc.predict(xtest)
Confusion Matrix
cm = np.array(confusion_matrix(ypred_test, ytest))
cm
array([[111,  60],
       [ 15,  64]], dtype=int64)
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="BuGn" ,fmt='g')
<AxesSubplot:>

Accuracy
print("Accuracy of training data:", accuracy_score(ytrain, ypred_train)*100)
ac2 = accuracy_score(ytest, ypred_test)*100
print("Accuracy of testing data:", ac2)
Accuracy of training data: 71.73333333333333
Accuracy of testing data: 70.0
K Nearest Neighbor
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(xtrain, ytrain)
KNeighborsClassifier(n_neighbors=1)
Prediction
ypred_train = knn.predict(xtrain)
ypred_test = knn.predict(xtest)
Confusion Matrix
cm = np.array(confusion_matrix(ypred_test, ytest))
cm
array([[101,  34],
       [ 25,  90]], dtype=int64)
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="Purples" ,fmt='g')
<AxesSubplot:>

Accuracy
print("Accuracy of training data:", accuracy_score(ytrain, ypred_train)*100)
ac3 = accuracy_score(ytest, ypred_test)*100
print("Accuracy of testing data:", ac3)
Accuracy of training data: 100.0
Accuracy of testing data: 76.4
Decission Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(xtrain, ytrain)
DecisionTreeClassifier(random_state=42)
Prediction
ypred_train = dt.predict(xtrain)
ypred_test = dt.predict(xtest)
Confusion Matrix
cm = np.array(confusion_matrix(ypred_test, ytest))
cm
array([[118,   8],
       [  8, 116]], dtype=int64)
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlOrRd" ,fmt='g')
<AxesSubplot:>

Prediction
print("Accuracy of training data:", accuracy_score(ytrain, ypred_train)*100)
ac4 = accuracy_score(ytest, ypred_test)*100
print("Accuracy of testing data:", ac4)
Accuracy of training data: 100.0
Accuracy of testing data: 93.60000000000001
Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(xtrain, ytrain)
RandomForestClassifier(random_state=42)
Prediction
ypred_train = rf.predict(xtrain)
ypred_test = rf.predict(xtest)
Confusion Matrix
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGn_r" ,fmt='g')
<AxesSubplot:>

Accuracy
print("Accuracy of training data:", accuracy_score(ytrain, ypred_train)*100)
ac5 = accuracy_score(ytest, ypred_test)*100
print("Accuracy of testing data:", ac5)
Accuracy of training data: 100.0
Accuracy of testing data: 94.8
Comparing Accuracy of Different Models
accuracy =  {ac1: 'Logistic Regression', ac2: 'SVM', ac3:'KNN', ac4:'Decission Tree', ac5: 'Random Forest'}
sns.set_style('darkgrid')
plt.figure(figsize=(14, 10))
model_accuracies = list(accuracy.values())
model_names = list(accuracy.keys())
sns.barplot(x=model_accuracies, y=model_names, palette='gist_rainbow')
<AxesSubplot:>
