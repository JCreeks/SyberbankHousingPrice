#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-5-24 下午4:54
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score


class KMeansImputeMissingData(object):
    """ Filling up missing data with KMeans cluster algorithm """

    def __init__(self, x, n_clusters, max_iter=10):
        """
            Args:
              X: An [n_samples, n_features] array of data to cluster.
              n_clusters: Number of clusters to form.
              max_iter: Maximum number of EM iterations to perform.
        """
        self.x = x
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def impute_missing_data(self):
        """Perform K-Means clustering on data with missing values.
        :return: kmeans_labels: kmeans 填充缺失值后聚类的labels
                 x_kmeans: kmeans 填充缺失值后的数据
                 centroids: kmeans 聚类后的中心
                 global_labels: 采用全局平均值填充缺失值后聚类的 labels
                 x_global_mean: 采用全局平均值填充缺失值后的数据
        """
        # Initialize missing values to their column means
        # 非数值型、正无穷和负无穷都认为是缺失数据
        missing = ~np.isfinite(self.x)
        mu = np.nanmean(self.x, axis=0, keepdims=True)  # 忽略 NaN，计算某一列的平均值
        x_kmeans = np.where(missing, mu, self.x)  # Return elements, either from x or y, depending on condition.
        x_global_mean = x_kmeans.copy()

        prev_centroids = None
        prev_labels = None
        global_labels = None
        kmeans_labels = None
        centroids = None
        for i in xrange(self.max_iter):
            if i > 0:
                # initialize KMeans with the previous set of centroids. this is much
                # faster and makes it easier to check convergence (since labels
                # won't be permuted on every iteration), but might be more prone to
                # getting stuck in local minima.
                cls = KMeans(self.n_clusters, init=prev_centroids)
            else:
                # do multiple random initializations in parallel
                cls = KMeans(self.n_clusters, n_jobs=-1)

            # perform clustering on the filled-in data
            kmeans_labels = cls.fit_predict(x_kmeans)
            if i == 0:
                global_labels = kmeans_labels.copy()
            centroids = cls.cluster_centers_

            # fill in the missing values based on their cluster centroids
            x_kmeans[missing] = centroids[kmeans_labels][missing]

            # when the labels have stopped changing then we have converged
            if i > 0 and np.all([kmeans_labels == prev_labels]):
                break

            prev_labels = kmeans_labels
            prev_centroids = cls.cluster_centers_

        return kmeans_labels, x_kmeans, centroids, global_labels, x_global_mean

    @staticmethod
    def report_performance(kmeans_labels, global_labels):
        """ 比较全局平均值填充和聚类后填充的互信息量，评估性能 """
        mutual_info = adjusted_mutual_info_score(kmeans_labels, global_labels)
        return mutual_info


def main():
    from sklearn.datasets import make_blobs

    def make_fake_data(fraction_missing, n_clusters=5, n_samples=5000,
                       n_features=3, seed=0):
        # complete data
        gen = np.random.RandomState(seed)
        X, true_labels = make_blobs(n_samples, n_features, n_clusters,
                                    random_state=gen)
        # with missing values
        missing = gen.rand(*X.shape) < fraction_missing
        Xm = np.where(missing, np.nan, X)
        return X, true_labels, Xm

    X, true_labels, Xm = make_fake_data(fraction_missing=0.1, n_clusters=6, seed=10)

    impute_model = KMeansImputeMissingData(Xm, n_clusters=5, max_iter=10)
    kmeans_labels, x_kmeans, centroids, global_labels, x_global_mean = impute_model.impute_missing_data()
    print '采用 KMeans 聚类填充缺失值与全局平均值填充缺失值的互信息量：', \
        impute_model.report_performance(kmeans_labels, global_labels)


if __name__ == '__main__':
    main()
