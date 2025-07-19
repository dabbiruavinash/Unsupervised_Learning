classify the Iris flowers data set using K-means Clustering model depending on petal length and petal widths of the flowers.

# clustering with k-means on iris flowers
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScalar
from matplotlib import pyplot as plt

# view the dataset
from sklearn.datasets import load_iris
irirs = load_iris()

# display the names of the attributes
dir(iris)

# display column names
iris.feature_names

# create dataframe with all columns
df = pd.DataFrame(iris.data, columns= iris.feature_names)
df

# Do scaling for petal length and width cols using MinMaxScalar
scaler = MinMaxScalar()
 
# fit the scale to petal length
scaler.fit(df[['petal length (cm)']])
df['petal length(cm)'] scaler.transform(df[['petal length (cm)']])

# fit the scale to petal width
scaler.fit(df[['petal width (cm)']])
df['petal width (cm)'] = scaler.transform(df[['petal width (cm)']])
df['petal width (cm)'] = scalar.transform(df[['petal width (cm)']])

# display the data frame after scaling of data
df.head()

# elbow plot to confirm the k value
sse = []
k_rng = range(1,10)
for k in k_rng:
      km = KMeans(n_clusters=k)
      km.fit(df[['petal length (cm)' , 'petal width (cm)']])
      sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)
# the above plot suggests k value is 3

# now fit the k-means clustering to petal length and width cols take the k value as 3 as suggested in the previous code
y_predicted = km.fit_predict(df[['petal length (cm)', 'petal width (cm)']])
y_predicted  # showing 0,1 and 2 clusters

# store the y_predicted values into cluster column in data frame
df['cluster'] = y_predicted
df.head()

# find cluster centers
km.cluster_centers_

# divide into 3 data frames depending on cluster numbers
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

# let us visualize the 3 groups. draw the scatter plots of each cluster with their centers.
plt.scatter(df1['petal length (cm) '].df1['petal width (cm)'], color = 'green', marker='+')
plt.scatter(df2['petal length (cm) '].df2['petal width (cm)'], color = 'red', marker = '+')
plt.scatter(df3['petal length (cm) '].df3['petal width (cm)'], color ='blue', marker = '+')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='purple', marker='*', label='centroid')
plt.xlabel('petal length')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend()