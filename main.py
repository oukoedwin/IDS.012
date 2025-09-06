import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy import stats
import seaborn as sns


# "./data/generators.csv"
extended_df = pd.read_csv("./data/generators.csv", header=0, index_col=0)

# remove these columns in a new copy of extended_df called df: "Winter-Summer Bid Diff", "Owner Generator Count"
df = extended_df.drop(columns=["Winter-Summer Bid Diff", "Owner Generator Count"])

## Rows: examples, columns: features 

### Preprocessing
numerical_columns = df.columns[df.dtypes != bool].tolist()
binary_columns = df.columns[df.dtypes == bool].tolist()
# binary_columns = ['Low Bidder', 'Diurnal Bidder', 'Reservation Market Bidder', 'Maximum Daily Energy Bidder']

df[binary_columns] = df[binary_columns].astype(int)

## Standardizing Continuous Data
# Remove zero-variance columns
zero_variance_columns = [col for col in numerical_columns if df[col].var() == 0]
df = df.drop(columns=zero_variance_columns)
numerical_columns = [col for col in numerical_columns if col not in set(zero_variance_columns)]

bad_rows = df[numerical_columns].isna().any(axis=1).sum()
df[numerical_columns] = df[numerical_columns].replace([np.inf, -np.inf], np.nan)
df_clean = df.dropna(subset=numerical_columns).copy()

df = df_clean

# scaler = StandardScaler()
scaler = MinMaxScaler(feature_range=(0, 1))
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

### Clustering
## 1. K-means
def kmeans(df, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    y_kmeans = kmeans.fit_predict(df)
    return y_kmeans

## 2. DBSCAN
def dbscan(df, eps=0.5, min_samples=5):
    # Focuses on regions of high point density, identifying clusters as regions of high
    # point density surrounded by regions of lower point density
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    y_dbscan = dbscan.fit_predict(df)
    return y_dbscan

def create_plot(y_labels, results, axes_names, title, plot_name):
    fig = plt.figure(figsize=(8, 6))
    num_components = len(axes_names)
    
    # Updated colormap selection (modern Matplotlib syntax)
    n_clusters = len(np.unique(y_labels))
    cmap = plt.colormaps.get_cmap('tab10').resampled(n_clusters)
    # Alternative options:
    # cmap = plt.get_cmap('tab10', n_clusters)  # pyplot version
    # cmap = plt.colormaps['tab10'].resampled(n_clusters)  # Direct access
    
    if num_components == 2:
        ax = fig.add_subplot(111)
        sc = ax.scatter(results[0], results[1], 
                        c=y_labels, cmap=cmap, alpha=0.6,
                        vmin=-0.5, vmax=n_clusters-0.5)
    elif num_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(results[0], results[1], results[2],
                        c=y_labels, cmap=cmap, alpha=0.6,
                        vmin=-0.5, vmax=n_clusters-0.5)
        ax.set_zlabel(axes_names[2])

    if title:
        ax.set_title(title)
    ax.set_xlabel(axes_names[0])
    ax.set_ylabel(axes_names[1])
    
    cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(n_clusters))
    cbar.set_label('Cluster')
    
    if plot_name:
        plt.savefig(plot_name, bbox_inches='tight', dpi=300)
    else:
        plt.show()

def pca(df, y_labels, n_components=2, title='', plot_name=''):
    # PCA
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(df)

    if n_components == 2:
        results = [pca_results[:, 0], pca_results[:, 1]]
        axes_names = ['Principal Component 1', 'Principal Component 2']
    elif n_components == 3:
        results = [pca_results[:, 0], pca_results[:, 1], pca_results[:, 2]]
        axes_names = ['Principal Component 1', 'Principal Component 2', 'Principal Component 3']

    create_plot(y_labels, results, axes_names, title, plot_name)

# tsne
def tsne(df, y_labels, n_components=2, title='', plot_name=''):
    tsne = TSNE(n_components=n_components, random_state=0)
    tsne_results = tsne.fit_transform(df)
    tsne1, tsne2, tsne3 = None, None, None
    if n_components == 2:
        tsne1 = tsne_results[:,0]
        tsne2 = tsne_results[:,1]
    elif n_components == 3:
        tsne1 = tsne_results[:,0]
        tsne2 = tsne_results[:,1]
        tsne3 = tsne_results[:,2]

    if n_components == 2:
        results = [tsne_results[:,0], tsne_results[:,1]]
        axes_names = ['t-SNE Component 1', 't-SNE Component 2']
    elif n_components == 3:
        results = [tsne_results[:,0], tsne_results[:,1], tsne_results[:,2]]
        axes_names = ['t-SNE Component 1', 't-SNE Component 2', 't-SNE Component 3']
    create_plot(y_labels, results, axes_names, title, plot_name)

def elbow_test(df):
    cluster_range = range(1, 15)
    scores = []
    for i in cluster_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df)
        scores.append(kmeans.inertia_)
    # plot
    plt.figure(figsize=(8, 6))
    plt.plot(cluster_range, scores)
    plt.title('Elbow Test for K-means Clustering')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-cluster Sum of Squares')
    plt.savefig('./plots/kmeans_elbow_test.png')
    # We found that num_clusters = 10 worked best

def elbow_test_db_scan(df, par="epsilon"):
    if par == "epsilon":
        # Reasonable range: 0.1 - 2.0
        cluster_range = [i for i in np.arange(0.1, 2.0, 0.1)]
    elif par == "minimum_samples":
        cluster_range = [i for i in range(5, 20)]   
    else:
        raise ValueError("par must be 'epsilon' or 'minimum_samples'")
    scores = []
    if par == "epsilon":
        for i in cluster_range:
            y_labels = DBSCAN(eps=i, min_samples=12).fit_predict(df)
            if len(y_labels[y_labels == -1]) > len(df) / 2:
                scores.append(0)
                continue
            if len(np.unique(y_labels[y_labels != -1])) > 1: 
                scores.append(silhouette_score(df[y_labels != -1], y_labels[y_labels != -1]))
            else:
                scores.append(0)  
    elif par == "minimum_samples":
        for i in cluster_range:
            y_labels = DBSCAN(eps=0.7, min_samples=i).fit_predict(df)
            if len(np.unique(y_labels)) > 1:
                scores.append(silhouette_score(df, y_labels))
            else:
                scores.append(0)

    plt.figure(figsize=(8, 6))
    plt.plot(cluster_range, scores)
    par_name = " ".join(par.split("_")).title()
    plt.title(f'Silhouette Score for Different Values of {par_name}')
    plt.xlabel(par_name)
    plt.grid(True)
    plt.ylabel('Silhouette Score')
    plt.savefig(f'./plots/dbscan_clustering_quality_test_{par}.png')
    ## Good values eps=0.1, min_samples=8

# elbow_test(df)
# elbow_test_db_scan(df, par="epsilon")
# elbow_test_db_scan(df, par="minimum_samples")

# KMeans Clustering
ykmeans = kmeans(df, num_clusters=10)
# print("kmeans silhouette score: ", silhouette_score(df[ykmeans != -1], ykmeans[ykmeans != -1]))
# Silhouette score: 0.536903052985577
# [120, 91, 50, 41, 34, 32, 29, 27, 17, 5]
# print("KMEANS:", pd.Series(ykmeans).value_counts())


## Visualizations
# ## Visualize kmeans
# pca(df, ykmeans, n_components=3, plot_name='./plots/kmeans_pca_3d.png')
# tsne(df, ykmeans, n_components=3, plot_name='./plots/kmeans_tsne_3d.png')

# # # DBSCAN Clustering
ydbscan = dbscan(df, eps=0.4, min_samples=8)
# Silhouette score: 0.54596
# [187, 137, 38, 34, 27, 23] # 23 unclassified
# print("dbscan silhouette score: ", silhouette_score(df[ydbscan != -1], ydbscan[ydbscan != -1]))
# print("DBSCAN:",pd.Series(ydbscan).value_counts())

# ## Visualize DBSCAN
# pca(df, ydbscan, n_components=2, plot_name='./plots/dbscan_pca.png')
# tsne(df, ydbscan, n_components=2, plot_name='./plots/dbscan_tsne.png')

## Hypothesis Testing (use extended_df): add two columns to a copy of df for "kmeans_cluster" and "dbscan_cluster" and fill them
# with the cluster labels from kmeans and dbscan respectively
# extended_df_clean = extended_df.loc[df.index].copy()[['Winter-Summer Bid Diff', 'Owner Generator Count']]  # keep rows that survived cleaning
extended_df_clean = extended_df.loc[df.index].copy()
extended_df_clean['kmeans_cluster'] = ykmeans
extended_df_clean['dbscan_cluster'] = ydbscan

## A Distribution of ownership of generators i.e number of generators owned by each firm
# Unique counts of owner generator count: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 23]

## Hypothesis 1: 
# Some clusters of generators (i.e., natural gas fuel-types => use predicted cluster labels) have higher bidding prices 
# than others during the winter than in the summer.
# **** H0: There is no difference in winter-summer bid differences across generator clusters.
# **** H1: At least one cluster has significantly different winter-summer bid differences compared to others.
def hypothesis1_qqplot(extended_df_clean, idx=0):
    # Get the cluster with most samples
    nth_largest_cluster = extended_df_clean['kmeans_cluster'].value_counts().sort_values(ascending=False).index[idx]
    print(f"Largest cluster has {extended_df_clean[extended_df_clean['kmeans_cluster'] == nth_largest_cluster].shape[0]} samples")
    
    # 1. Check normality with QQ plot for largest cluster
    plt.figure(figsize=(10, 6))
    cluster_data = extended_df_clean[extended_df_clean['kmeans_cluster'] == nth_largest_cluster]['Winter-Summer Bid Diff']
    stats.probplot(cluster_data, dist="norm", plot=plt)
    plt.ylabel("Sample Quantiles")
    plt.title(f'QQ Plot for Winter-Summer Bid Differences for Generators in the Largest Cluster')
    plt.savefig('./hypothesis_plots/qq_plot_2_largest_cluster.png')

# hypothesis1_qqplot(extended_df_clean, idx=1)
    
def hypothesis1_test(extended_df_clean, significance_level=0.05):
    # Shapiro-Wilk test for normality for all clusters
    normal_dist = True
    for cluster in extended_df_clean['kmeans_cluster'].unique():
        cluster_data = extended_df_clean[extended_df_clean['kmeans_cluster'] == cluster]['Winter-Summer Bid Diff'].dropna()
        if len(cluster_data) < 15:
            continue
        stat, p = stats.shapiro(cluster_data)
        # print(f"Cluster {cluster}: W = {stat:.7f}, p = {p:.7f}")
        if p < significance_level:
            normal_dist = False

    # 2. Perform appropriate overall test
    if normal_dist and len(extended_df_clean['kmeans_cluster'].unique()) > 2:
        print("\nData appears normally distributed - using ANOVA")
        # One-way ANOVA
        groups = []
        for cluster in extended_df_clean['kmeans_cluster']:
            # skip clusters with less than 15 samples
            val = extended_df_clean[extended_df_clean['kmeans_cluster'] == cluster]['Winter-Summer Bid Diff'].dropna()
            if len(val) < 15:
                continue
            groups.append(val)
        f_val, p_val = stats.f_oneway(*groups)
        print(f"ANOVA results: F = {f_val:.7f}, p = {p_val:.7f}")
    else:
        print("\nData not normally distributed - using Kruskal-Wallis")
        # Kruskal-Wallis test
        groups = []
        for cluster in extended_df_clean['kmeans_cluster'].unique():
            val = extended_df_clean[extended_df_clean['kmeans_cluster'] == cluster]['Winter-Summer Bid Diff'].dropna()
            if len(val) < 15:
                continue
            groups.append(val)
        h_val, p_val = stats.kruskal(*groups)
        print(f"Kruskal-Wallis results: H = {h_val:.7f}, p = {p_val:.7f}")

    if p_val >= significance_level:
        print("\nOverall test not significant")
        return

    print("\nOverall test significant - performing pairwise comparisons")
    # Only clusters with more than 15 samples

    # clusters = extended_df_clean[extended_df_clean['kmeans_cluster'].value_counts() >= 15]['kmeans_cluster'].unique()
    cluster_counts = extended_df_clean['kmeans_cluster'].value_counts()
    clusters = cluster_counts[cluster_counts >= 15].index.to_list()

    # Pairwise t-tests (with Bonferroni correction)
    if normal_dist:
        print("\nPairwise t-tests with Bonferroni correction:")
        n_comparisons = len(clusters) * (len(clusters) - 1) / 2
        alpha_corrected = significance_level / n_comparisons
        
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                group1 = extended_df_clean[extended_df_clean['kmeans_cluster'] == clusters[i]]['Winter-Summer Bid Diff']
                group2 = extended_df_clean[extended_df_clean['kmeans_cluster'] == clusters[j]]['Winter-Summer Bid Diff']
                t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                print(f"Cluster {clusters[i]} vs {clusters[j]}: t = {t_stat:.7f}, p = {p_val:.7f}", 
                        "*" if p_val < alpha_corrected else "")
    else: # Pairwise Mann-Whitney U tests if not normal
        print("\nPairwise Mann-Whitney U tests with Bonferroni correction:")
        n_comparisons = len(clusters) * (len(clusters) - 1) / 2 
        alpha_corrected = significance_level / n_comparisons
        
        for i in range(len(clusters)):
            # skip clusters with less than 15 samples
            group1 = extended_df_clean[extended_df_clean['kmeans_cluster'] == clusters[i]]['Winter-Summer Bid Diff'].dropna()
            if len(group1) < 15:
                continue
            for j in range(i+1, len(clusters)):
                group2 = extended_df_clean[extended_df_clean['kmeans_cluster'] == clusters[j]]['Winter-Summer Bid Diff'].dropna()
                if len(group2) < 15:
                    continue
                u_stat, p_val = stats.mannwhitneyu(group1, group2)
                print(f"Cluster {clusters[i]} vs {clusters[j]}: U = {u_stat:.7f}, p = {p_val:.7f}", 
                        "*" if p_val < alpha_corrected else "")

hypothesis1_test(extended_df_clean)


def hypothesis1_plot(extended_df_clean):
    plt.figure(figsize=(12, 10))  
    sns.boxplot(x='kmeans_cluster', y='Winter-Summer Bid Diff', data=extended_df_clean)
    plt.title('Winter-Summer Bid Differences by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Winter - Summer Average Bid Difference ($/MWh)')
    plt.savefig('./hypothesis_plots/cluster_winter_summer_bid_diff.png')

# hypothesis1_plot(extended_df_clean)

## Hypothesis 2: Generators owned by firms with substantial of generation capacity tend to bid relatively higher 
# prices than their respective clusters (i.e generators of the same type or predicted label).
# **** H0: There is no relationship between owner generator count and bidding prices within clusters.
# **** H1: Generators owned by firms with more generators bid higher prices within their clusters.

# df["kmeans_cluster"] = ykmeans
# df["dbscan_cluster"] = ydbscan
# df["Owner Generator Count"] = extended_df_clean["Owner Generator Count"]
# df["Winter-Summer Bid Diff"] = extended_df_clean["Winter-Summer Bid Diff"]
def hypothesis2_test(df):
    # Create groups based on owner generator count (e.g. above/below median)
    for cluster in df['kmeans_cluster'].unique():
        # only consider clusters with more than 15 samples
        if df['kmeans_cluster'].value_counts()[cluster] < 15:
            continue
        cluster_data = df[(df['kmeans_cluster'] == cluster)]
        median_count = cluster_data['Owner Generator Count'].median()
        high_count = cluster_data[cluster_data['Owner Generator Count'] > median_count]['Average Bid Start']
        low_count = cluster_data[cluster_data['Owner Generator Count'] <= median_count]['Average Bid Start']
        
        # t-test if normally distributed
        t_stat, p_val = stats.ttest_ind(high_count, low_count, equal_var=False)
        print(f"Cluster {cluster} - High vs Low Owner Count:")
        print(f"  High count mean: {high_count.mean():.2f}, Low count mean: {low_count.mean():.2f}")
        print(f"  t-statistic: {t_stat:.2f}, p-value: {p_val:.4f}")

# hypothesis2_test(df)


## NOTES
# nuclear, hydro, wind, solar, battery
# Discussion: number of clusters in the real, number of clusters predicted by our cluster quality score
# Paul expects n_clusters = 7, or 8
# Elbow test: cluster quality score against num_clusters
## Hypothesis Testing
# - All analysis done on clusters with more than 15 samples (5 clusters)
# - Checked for normal distribution in hypothesis 1. Made a qq-plot for the 3 largest clusters. 
#   Used Shapiro-Wilk test to check normality for all clusters with more than 15 elements in them => some clusters are not normal
# - Kruskal-Wallis test: Does not require normal distribution of the data; compares the medians of multiple independent groups; global view of all clusters
# - Wilcoxon rank-sum or Mann-Whitney U test for non-normal clusters: More granular view of where the differences come from. Data drawn from independent populations. 
#   Rank-test => tests whether one cluster tends to have higher bid prices than another cluster.
# - Bonferroni correction to correct each pairwise Wilcoxon rank-sum tests
# - Exploring different significance levels: 0.05, 0.03, 0.01
# - Weak assumption of independence for generators (we know generators owned by the same firm are not independent)






