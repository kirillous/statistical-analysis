import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.stats import shapiro
from scipy.stats import kruskal

data = pd.read_excel("Assignment8_data.xlsx").set_axis(["X1", "X2", "Y1", "Y2"], axis=1)

# matrix pair plot
sns.pairplot(data)
plt.show()

# box plot
sns.boxplot(data)
plt.show()

# correlation matrix
corr_matrix = data.corr()
print("\nCorrelation matrix:")
print(corr_matrix)
sns.heatmap(corr_matrix,annot=True,fmt=".3f")
plt.show()

# covariance matrix
cov_matrix = data.cov()
sns.heatmap(cov_matrix,annot=True,fmt=".2f")
plt.show()

# Scatter plots
# X1 vs Y1
plt.scatter(data["X1"].values, data["Y1"].values)
plt.title("X1 vs Y1 Scatter plot")
plt.xlabel("X1")
plt.ylabel("Y1")
plt.grid(True)
plt.show()
# X1 vs Y2
plt.scatter(data["X1"].values, data["Y2"].values)
plt.title("X1 vs Y2 Scatter plot")
plt.xlabel("X1")
plt.ylabel("Y2")
plt.grid(True)
plt.show()
# Y1 vs Y2
plt.scatter(data["Y1"].values, data["Y2"].values)
plt.title("Y1 vs Y2 Scatter plot")
plt.xlabel("Y1")
plt.ylabel("Y2")
plt.grid(True)
plt.show()

# 3D scatter plot of X1, Y1, Y2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data["X1"], data["Y1"], data["Y2"])
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.set_title('3D Scatter plot')
plt.show()

# LINEAR REGRESSIONS
# X1 vs Y1
X = data['X1'].values.reshape(-1, 1)
Y = data['Y1'].values
model = LinearRegression()
model.fit(X, Y)
m = model.coef_[0]
b = model.intercept_
print("X1 vs Y1: y = {:.3f}x + {:.3f}".format(m, b))
# X1 vs Y1
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Linear regression')
plt.title('X1 vs Y1 linear regression')
plt.xlabel('X1')
plt.ylabel('Y1')
plt.legend()
plt.grid(True)
plt.show()
# X1 vs Y2
X = data['X1'].values.reshape(-1, 1)
Y = data['Y2'].values
model = LinearRegression()
model.fit(X, Y)
m = model.coef_[0]
b = model.intercept_
print("X1 vs Y2: y = {:.3f}x + {:.3f}".format(m, b))
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Linear regression')
plt.title('X1 vs Y2 linear regression')
plt.xlabel('X1')
plt.ylabel('Y2')
plt.legend()
plt.grid(True)
plt.show()
# Y1 vs Y2
X = data['Y1'].values.reshape(-1, 1)
Y = data['Y2'].values
model = LinearRegression()
model.fit(X, Y)
m = model.coef_[0]
b = model.intercept_
print("X1 vs Y2: y = {:.3f}x + {:.3f}".format(m, b))
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Linear regression')
plt.title('Y1 vs Y2 linear regression')
plt.xlabel('Y1')
plt.ylabel('Y2')
plt.legend()
plt.grid(True)
plt.show()

# PCA
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
total_variance = np.sum(eigenvalues)
percent_contributions = (eigenvalues / total_variance) * 100
print("\nEigenvalues:")
print(eigenvalues)
print("\nPercent contribution to variance:")
for i, percent in enumerate(percent_contributions):
    print(f"PC{i + 1}: {percent:.2f}%")
eigenvector_pc1 = eigenvectors[:, 0]
eigenvector_pc2 = eigenvectors[:, 1]

print("\nEigenvector for PC1:")
print(eigenvector_pc1)
print("\nEigenvector for PC2:")
print(eigenvector_pc2)

pc1_values = np.dot(data, eigenvector_pc1)
pc2_values = np.dot(data, eigenvector_pc2)
pca_data = pd.DataFrame({"pc1":pc1_values,"pc2":pc2_values})
# PCA plot
plt.figure(figsize=(8, 6))
plt.scatter(pc1_values, pc2_values, c='b', alpha=0.5)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)
plt.show()

# cluster analysis on PCA data
pca_num_data = pca_data[["pc1","pc2"]].values
kmeans = KMeans(n_clusters=3)
kmeans.fit(pca_num_data)
cluster_labels = kmeans.labels_
pca_data["Cluster"] = cluster_labels
pca_data["Cluster"] += 1
# cluster plot
colors = ["red", "green", "blue"] 
plt.figure(figsize=(8, 6))
for cluster_id, group in pca_data.groupby("Cluster"):
    plt.scatter(group["pc1"], group["pc2"], color=colors[cluster_id-1], alpha=0.5, label=f"Cluster {cluster_id}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.legend()
plt.show()

# CONTINUATION
# Normality tests
cluster_1 = pca_data[pca_data["Cluster"] == 1]
cluster_2 = pca_data[pca_data["Cluster"] == 2]
cluster_3 = pca_data[pca_data["Cluster"] == 3]

clusters = [cluster_1, cluster_2, cluster_3]

normality_matrix = []
alpha = 0.05
for i, cluster in enumerate(clusters, start=1):
    cluster_results = []
    for j in range(1,3):
        pc_name = f"pc{j}"
        statistic, p_value = shapiro(cluster[pc_name])
        is_normal = p_value > alpha
        cluster_results.append(is_normal)
    normality_matrix.append(cluster_results)
normality_matrix = pd.DataFrame(normality_matrix, columns=["PC1", "PC2"], index=["Cluster 1", "Cluster 2", "Cluster 3"])
print(normality_matrix)

# non parametric test for differernce in clusters (Kruskal-Wallis test)
kw_stat_both, p_value_both = kruskal(
    cluster_1["pc1"], cluster_2["pc1"], cluster_3["pc1"],
    cluster_1["pc2"], cluster_2["pc2"], cluster_3["pc2"]
)

print("\nKruskal-Wallis Test Results for PC1 and PC2:")
print(f"Statistic = {kw_stat_both}, p-value = {p_value_both}")
if p_value_both < 0.05:
    print("Clusters are differernt")
else:
    print("Clusters are NOT different")




