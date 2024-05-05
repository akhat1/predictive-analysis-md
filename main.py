import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans


# Load the dataset
df = pd.read_csv('online_shoppers_intention.csv')

# Display the first few rows of the DataFrame
print(df.head())

# get the size of the dataset 
print(df.shape)

print(df.head(10))

print(df.describe())

mv=df.isnull().sum()
print(mv)


sns.set(style="darkgrid")
plt.figure(figsize=(8,5))
total = float(len(df))
ax = sns.countplot(x="Revenue", data=df)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
x = p.get_x() + p.get_width()
y = p.get_height()
ax.annotate(percentage, (x, y),ha='center')
plt.show()



df['VisitorType'].value_counts()
sns.set(style="whitegrid")
plt.figure(figsize=(8,5))
total = float(len(df))
ax = sns.countplot(x="VisitorType", data=df, palette='inferno')
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
x = p.get_x() + p.get_width()
y = p.get_height()
ax.annotate(percentage, (x, y),ha='center')
plt.show()



x, y = 'VisitorType', 'Weekend'
df1 = df.groupby(x)[y].value_counts(normalize=True).mul(100).rename('percent').reset_index()




# Create the plot with improved visualization settings
g = sns.catplot(x=x, y='percent', hue=y, kind='bar', data=df1, palette='inferno', height=6, aspect=1.5)
g.ax.set_ylim(0, 105)  # Increase ylim to make room for annotations

# Annotate the percentages on the bars
for p in g.ax.patches:
    txt = '{:.2f}%'.format(p.get_height())
    txt_x = p.get_x() + p.get_width() / 2  # Adjust x position to center the text
    txt_y = p.get_height() + 1             # Adjust y position to lift the text above the bar
    g.ax.text(txt_x, txt_y, txt, ha='center')

plt.show()




x='TrafficType'
y= 'Revenue'
df1 = df.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()
g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1, palette='inferno')
g.ax.set_ylim(0,100)
for p in g.ax.patches:
    txt = '{:.2f}%'.format(p.get_height())
    txt_x = p.get_x() + p.get_width() / 2
    txt_y = p.get_height() + 1
    g.ax.text(txt_x, txt_y, txt, ha='center')
plt.show()



plt.hist(df['TrafficType'],color='royalblue',rwidth=0.85)
plt.title('Distribution of diff Traffic',fontsize = 30)
plt.xlabel('TrafficType Codes', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()




plt.hist(df['Region'], color='royalblue',rwidth=0.85)
plt.title('Distribution of Customers',fontsize = 30)
plt.xlabel('Region Codes', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()



plt.hist(df['OperatingSystems'], color='royalblue', rwidth=0.85)
plt.title('Distribution of Customers',fontsize = 30)
plt.xlabel('OperatingSystems', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()



plt.hist(df['Month'], color='royalblue', rwidth=0.85)
plt.title('Distribution of Customers',fontsize = 30)
plt.xlabel('Month', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()





sns.stripplot(x='Revenue', y='PageValues', data=df, palette='inferno')
plt.show()




sns.stripplot(x='Revenue', y='BounceRates', data=df,palette='inferno')
plt.show()




df1 = pd.crosstab(df['TrafficType'], df['Revenue'])
# Normalize the data to sum to 1 (100%) for each group
df_norm = df1.div(df1.sum(1).astype(float), axis=0)

# Plotting with custom colors
df_norm.plot(kind='bar', stacked=True, color=['silver', 'purple'])
plt.title('Traffic Type vs Revenue', fontsize=30)
plt.show()



ax4 = sns.countplot(x='Region', hue='Revenue', data=df, palette='inferno')
ax4.set_title('Distribution of Revenue by Region')
ax4.set_xlabel('Region')
ax4.set_ylabel('Count')
plt.show()



sns.lmplot(x = 'Administrative', y = 'Informational', data = df,scatter_kws={'color': 'purple'}, line_kws={'color': 'purple'},  x_jitter = 0.05)
plt.show()




sns.boxplot(x = df['Month'], y = df['PageValues'], hue = df['Revenue'], palette = 'inferno')
plt.title('Mon. vs PageValues w.r.t. Rev.', fontsize = 30)
plt.show()



# visitor type vs BounceRates w.r.t revenue
sns.boxplot(x = df['VisitorType'], y = df['BounceRates'], hue = df['Revenue'], palette = 'plasma')
plt.title('Visitors vs BounceRates w.r.t. Rev.', fontsize = 30)
plt.show()



# Mon vs ExitRates w.r.t revenue
sns.boxplot(x = df['Month'], y = df['ExitRates'], hue = df['Revenue'], palette = 'plasma')
plt.title('Month vs ExitRates w.r.t. Rev.', fontsize = 30)
plt.show()




# Preparing the dataset
x = df.iloc[:, [1, 6]].values  # Adjust column indices as needed

# Calculating the WCSS (Within-Cluster Sum of Square)
wcss = []
for i in range(1, 11):
    algorithm = 'elkan' if i > 1 else 'lloyd'
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0, algorithm=algorithm, tol=0.001)
    km.fit(x)
    wcss.append(km.inertia_)

# Creating a figure and axis object
fig, ax = plt.subplots(figsize=(15, 7))

# Plotting the Elbow Method graph on the defined axis
ax.plot(range(1, 11), wcss)
ax.grid(True)
ax.set_title('The Elbow Method', fontsize=20)
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('WCSS')

# Display the plot
plt.show()


# Initializing and fitting the KMeans algorithm
km = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means = km.fit_predict(x)

# Plotting the clusters
plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s=100, c='red', label='Un-interested Customers')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s=100, c='yellow', label='General Customers')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s=100, c='green', label='Target Customers')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=50, c='blue', label='Centroids')

# Adding titles and labels
plt.title('Cluster Visualization', fontsize=20)
plt.xlabel('Administrative Duration')  # Assuming column 1 is Administrative Duration
plt.ylabel('Informational Duration')   # Assuming column 6 is Informational Duration
plt.legend()

# Displaying the plot
plt.grid(True)
plt.show()




# Assuming df is loaded and has the correct columns indexed at [3, 6]
x = df.iloc[:, [3, 6]].values

wcss = []
for i in range(1, 11):
    algorithm = 'lloyd' if i == 1 else 'elkan'
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0, algorithm=algorithm, tol=0.001)
    km.fit(x)
    wcss.append(km.inertia_)

# Explicit figure creation
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(range(1, 11), wcss)
ax.grid(True)
ax.set_title('The Elbow Method', fontsize=20)
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('WCSS')

plt.show()


km = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 600, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)
plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'red', label = 'Un-interested Customers')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'Target Customers')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')
plt.title('Informational Duration vs Bounce Rates', fontsize = 20)
plt.grid()
plt.xlabel('Informational Duration')
plt.ylabel('Bounce Rates')
plt.legend()
plt.show()



# Region vs Traffic Type
# Preparing the dataset by selecting the relevant columns
x = df.iloc[:, [13, 14]].values  # Adjust indices according to your DataFrame

# Initializing the list to store the within-cluster sum of squares (WCSS)
wcss = []

# Looping from 1 to 10 to find the optimal number of clusters
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0, algorithm='elkan', tol=0.001)
    km.fit(x)
    labels = km.labels_
    wcss.append(km.inertia_)

# Setting the figure size for the plot
plt.rcParams['figure.figsize'] = (15, 7)

# Plotting the WCSS values to observe the 'elbow'
plt.plot(range(1, 11), wcss)
plt.grid(True)
plt.tight_layout()
plt.title('The Elbow Method', fontsize=20)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
#plt.show()


# Initialize and fit the KMeans algorithm with 2 clusters
km = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means = km.fit_predict(x)

# Plotting clusters
plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s=100, c='red', label='Un-interested Customers')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s=100, c='yellow', label='Target Customers')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=50, c='blue', label='Centroids')

# Adding titles and labels
plt.title('Region vs Traffic Type', fontsize=20)
plt.xlabel('Region')  # Make sure this corresponds to the correct feature
plt.ylabel('Traffic Type')  # Ensure this is the correct label
plt.legend()

# Displaying the plot
plt.grid(True)
plt.show()


# one hot encoding
data1 = pd.get_dummies(df)
data1.columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Revenue'] = le.fit_transform(df['Revenue'])
df['Revenue'].value_counts()
# getting dependent and independent variables
x=data1
# removing the target column revenue from 
x = x.drop(['Revenue'], axis = 1)
y = data1['Revenue']
# checking the shapes
print("Shape of x:", x.shape)
print("Shape of y:", y.shape)


# splitting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
# checking the shape
print("Shape of x_train :", x_train.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_test :", y_test.shape)

# MODELLING
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# evaluating the model
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize'] = (6, 6)
sns.heatmap(cm ,annot = True)
# classification report
cr = classification_report(y_test, y_pred)
print(cr)
plt.show()



# Generating the confusion matrix
cm = confusion_matrix(y, model.predict(x))

# Setting up the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # cmap for coloring

# Remove grid lines
ax.grid(False)

# Setting the ticks and labels on both axes
ax.xaxis.set_ticks([0, 1])
ax.xaxis.set_ticklabels(['Predicted 0s', 'Predicted 1s'])
ax.yaxis.set_ticks([0, 1])
ax.yaxis.set_ticklabels(['Actual 0s', 'Actual 1s'])

# Inverting y-axis to display the first row on top
ax.set_ylim(1.5, -0.5)

# Looping over data dimensions and creating text annotations
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')

# Display the plot
plt.show()


from sklearn.metrics import roc_curve, auc

# Assuming 'model', 'x_test', and 'y_test' are already defined
# Make sure your model can predict probabilities
y_scores = model.predict_proba(x_test)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()



df = pd.DataFrame(y_pred, columns=["Revenue"])

# Saving the DataFrame to a CSV file
df.to_csv('predicted_revenue.csv', index=False)



from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(solver='liblinear', random_state=0)
model1.fit(x_train, y_train)
y_pred1 = model1.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix : \n", cm)

# Compute the confusion matrix
cm = confusion_matrix(y, model.predict(x))

# Create a figure and an axes object
fig, ax = plt.subplots(figsize=(8, 8))

# Display the confusion matrix
ax.imshow(cm, interpolation='nearest', cmap='Blues')  # Added color map for better visualization

# Remove grid
ax.grid(False)

# Setting the ticks and tick labels on both axes
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))

# Setting the limit for the y-axis
ax.set_ylim(1.5, -0.5)

# Loop through the data dimensions and create text annotations.
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')

# Display the plot
plt.show()



# classification report
cr1 = classification_report(y_test, y_pred1)
print(cr1)


from sklearn.metrics import roc_curve, auc
y_scores = model1.predict_proba(x_test)[:, 1]  # Adjust this if your model uses a different method

# Calculate ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


df1 = pd.DataFrame(y_pred1, columns=["Revenue"])
df1.to_csv('predicted1_revenue.csv', index=False)

from sklearn.metrics import roc_curve, auc

# Get model predictions
y_scores_model = model.predict_proba(x_test)[:, 1] 
y_scores_model1 = model1.predict_proba(x_test)[:, 1]

# Calculate ROC curve and ROC area for model
fpr_model, tpr_model, _ = roc_curve(y_test, y_scores_model)
roc_auc_model = auc(fpr_model, tpr_model)

# Calculate ROC curve and ROC area for model1
fpr_model1, tpr_model1, _ = roc_curve(y_test, y_scores_model1)
roc_auc_model1 = auc(fpr_model1, tpr_model1)

# Creating a plot
plt.figure(figsize=(10, 8))
plt.plot(fpr_model, tpr_model, color='darkorange', lw=2, label=f'Model (area = {roc_auc_model:.2f})')
plt.plot(fpr_model1, tpr_model1, color='green', lw=2, label=f'Model1 (area = {roc_auc_model1:.2f})')

# Plotting the random chance line
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Setting the limits for the x and y axes
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# Adding labels, title, and a legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Displaying the plot
plt.show()

