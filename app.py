import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)

DATA_URL = (
    "CC GENERAL.csv"
)

st.title("Unsupervised Machine Learning for Customer Market Segmentation")
st.markdown("This application is a Streamlit dashboard that can be used "
            "to analyze Customer segmentation of A Bank's Credit card user's"
            "using K-Means Algorithm")

if st.sidebar.checkbox("Show Raw Data", False):
    st.subheader('Raw Data')
    st.write(data = pd.read_csv(DATA_URL))


creditcard_df = pd.read_csv(DATA_URL)
creditcard_df.loc[(creditcard_df['MINIMUM_PAYMENTS'].isnull() == True),'MINIMUM_PAYMENTS'] = creditcard_df['MINIMUM_PAYMENTS'].mean()
creditcard_df.loc[(creditcard_df['CREDIT_LIMIT'].isnull() == True),'CREDIT_LIMIT'] = creditcard_df['CREDIT_LIMIT'].mean()
creditcard_df.drop('CUST_ID', axis = 1, inplace = True)

st.header("Lets study the data more thoroughly using subplots")



fig = plt.figure(figsize = (10,50))
for i in range(len(creditcard_df.columns)):
    fig = plt.subplot(17,1,i+1)
    fig = sns.distplot(creditcard_df[creditcard_df.columns[i]], kde_kws={'color':'b', 'lw':5, 'label':'KDE'}, hist_kws={'color':'g'})
    fig = plt.title(creditcard_df.columns[i])
fig = plt.tight_layout()
st.pyplot(fig)

st.header("Lets plot the corelation Matrix")


correlations = creditcard_df.corr()
f = plt.figure(figsize = (20,10))
f = sns.heatmap(correlations, annot = True)
st.pyplot(f)

scaler = StandardScaler()
creditcard_df_scaled = scaler.fit_transform(creditcard_df)

st.header("Lets use Elbow method to find optimal value of k")

scores_1 = []
range_values = range(1,20)

for i in range_values:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(creditcard_df_scaled)
    scores_1.append(kmeans.inertia_)

x = plt.xlabel('no. of clusters')
x = plt.ylabel('error')
x = plt.plot(scores_1, 'bx-')
st.pyplot(x)

st.header("As interpreted from fig. We will take 8 as our value of k")
st.subheader("Lets apply kmeans with k = 8 ")

kmeans = KMeans(8)
kmeans.fit(creditcard_df_scaled)
labels = kmeans.labels_

cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [creditcard_df.columns])

cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [creditcard_df.columns])

st.header("This are the segmentations we get")

st.write(cluster_centers)

y_means = kmeans.fit_predict(creditcard_df_scaled)

creditcard_df_cluster = pd.concat([creditcard_df, pd.DataFrame({'cluster':labels})],axis =1)

st.header("Lets see how each varible varies in each Cluster")

for i in creditcard_df.columns:
    b=plt.figure(figsize= (35, 5))
    for j in range(8):
        b=plt.subplot(1,8,j+1)
        cluster = creditcard_df_cluster[creditcard_df_cluster['cluster'] ==j]
        b=cluster[i].hist(bins = 20)
        b=plt.title('{}  \nCluster {}'.format(i,j))
st.pyplot(b)

st.header("Lets Apply PCA to obtain a 2d plot")

pca = PCA(n_components=2)
principal_comp = pca.fit_transform(creditcard_df_scaled)

pca_df = pd.DataFrame(data = principal_comp, columns = ['pca1', 'pca2'])

pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis =1)

c=plt.figure(figsize=(10,10))

c, ax = sns.scatterplot(x='pca1', y='pca2', hue = 'cluster', data = pca_df, palette = ['red','green','blue','pink','yellow','purple','gray','black'])

st.pyplot(c)



st.header("Thus the customers have been segmented in 8 different types")
