Use K means machine learning model and classify the employee into various income groups or clusters.

# grouping the salaries with k-means clustering
import pandas as pd
from sklearn.preprocessing import MinMaxScalar
from matplotlib import pyplot as plt

# view the dataset
df = pd.read_csv("E:/test/income.csv")
df

# create scatter plot to see the groups or clusters
plt.scatter(df['Age'], df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')

# since 3 clusters are seen, let us use k means clustering
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted # 0,1,2---> there are 3 clusters

# add this cluster as another column
df['cluster'] = y_predicted
df.head()

# find the center coordinates of clusters
print(km.cluster_centers_)

# 0th col and 1st col represents x,y coordinates of the cluster centers
x = km.clusters_centers_[:,0] # all rows in 0th column
y = km.clusters_centers_[:,1] # all rows in 1st column

# separate the 3 clusters into 3 dataframes
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

# scatter plot the cluster
plt.scatter(df1.Age,df1['Income($)'],color='green', marker='o')
plt.scatter(df2.Age,df2['Income($)'],color='red', marker='d')
plt.scatter(df3.Age,df3['Income($)'], color='black', marker='s')

#scatter plot the cluster centers
plt.scatter(x,y,color='purple', marker='*', label='centroid')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()

# the cluster are not grouped correctly
# the reason is scaling of data is not done.
# MinMaxScaler can be used for proper scaling of data
scalar = MinMaxScalar()

# fit the scale to income
scalar.fit(df[['Income($)']])
df['Income($)'] = scalar.transform(df[['Income($)']])

# fit the scale to Age
scalar.fit(df[['Age']])
df['Age'] = scalar.transform(df[['Age']])

# display data frame after scaling
df.head()

# elbow plot to find correct value of k
# calculate SSE values for each k value from 1 to 9
sse = []
k_rng = range(1,10)
for k in k_rng:
      km = KMeans(n_clusters=k)
      km.fit(df[['Age','Income($)']])
      sse.append(km.inertia_) # inertia_ = sum of squared distance

# draw the plot between K and SSE values
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)
plt.show()

# the plot shows that there are 3 income groups