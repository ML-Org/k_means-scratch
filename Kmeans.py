import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from DBIndex import DBIndex
from sklearn.cluster import KMeans as KMEANS
from utils import visualize, normalize_columns

# benchmarking purpose only
from sklearn.metrics import davies_bouldin_score
import logging
import seaborn as sns
import pickle as pkl


# monkey path round to identity function
# done due to laziness to remove round function everywhere
_round = lambda x,y : x

class KMeans:

    def __init__(self, n_clusters, n_iter, random_state ):
        self.k = n_clusters
        self.n_iter= n_iter
        self.seed = random_state
        logging.basicConfig(level=logging.INFO)
        self.log = logging

    def init_centroids(self,data, k, seed):
        return data.sample(k, random_state=seed).values


    def get_membership(self, data, centroids, features):
        membership = {}
        labels = pd.DataFrame()
        labels["cluster_id"] = np.zeros(shape=(data.shape[0]),dtype=np.int64)
        #row_idx=0;
        for i, row in enumerate(data[features].values):
            # get the index with least distance and use that cluster as dict indices
            index = np.argmin(_round([np.linalg.norm(row - centroid) for centroid in centroids], 4))
            # paritition matrix contains centroid info about each point
            labels["cluster_id"].loc[i] = index
            cluster_index = tuple(centroids[index])
            if cluster_index in membership.keys():
                # if cluster is non empty append points
                membership[cluster_index].append(row)
            else:
                # if cluster is empty assign points
                membership[cluster_index] = [row]
            #i +=1
        return labels,membership



    def equality_check(self, centroids_old, centroids_new=[]):

        return np.array_equal(centroids_old, centroids_new)



    def converge(self, _iter, centroids_old, centroids_new):
        """
        checks if arrays changed or iterations exhausted
        """
        #return self.equality_check(centroids_old, centroids_new) or _iter == 0
        return _iter == 0


    def fit(self, data):
        centroids = self.init_centroids(data, self.k, seed=self.seed)
        #Compute the membership matrix
        _,membership_matrix = self.get_membership(data, centroids, data.columns)
        #recompute new centroids from the clusters
        # todo: attribute round
        centroids_new = [_round(np.average(membership_matrix[cluster], axis=0), 4) for cluster in membership_matrix]
        #Run the algorithm until it converges
        max_iter = self.n_iter
        centroids_old = centroids
        centroids_new = self.init_centroids(data, self.k, seed=self.seed + 100)
        for i in range(0,self.n_iter):
            _labels, membership_matrix = self.get_membership(data, centroids_old, data.columns)
            centroids_new = [np.average(membership_matrix[cluster], axis=0) for cluster in membership_matrix]
            if (np.array_equal(centroids_old, centroids_new)):
                break
            self.log.debug("###############")
            self.log.debug(self.n_iter - max_iter, centroids_new)
            self.log.debug("###############")
            centroids_old = centroids_new
            max_iter -= 1
            self.log.debug("iteration - {}".format(self.n_iter - max_iter))
        # while (not self.converge(max_iter, centroids_old, centroids_new)):
        #     centroids_old = centroids_new
        #     _labels,membership_matrix = self.get_membership(data, centroids_old, data.columns)
        #     centroids_new = [np.average(membership_matrix[cluster], axis=0) for cluster in membership_matrix]
        #     print("###############")
        #     print(self.n_iter - max_iter, centroids_new)
        #     print("###############")
        #     max_iter -= 1
        #     print("iteration - {}".format(self.n_iter - max_iter))
        final_centroids = list(membership_matrix)
        return final_centroids, membership_matrix, _labels


if __name__ == "__main__":
    # load data
    bsom_data = pd.read_csv("BSOM_DataSet_revised.csv")
    requried_cols = ["all_NBME_avg_n4", "all_PIs_avg_n131", "HD_final"]
    bsom_3_features = normalize_columns(bsom_data[requried_cols])

    k_means = KMeans(n_clusters=3, n_iter=100, random_state=120)
    centroids,membership_matrix, labels = k_means.fit(bsom_3_features)
    plt, ax=visualize(centroids, membership_matrix, labels=bsom_3_features.columns,save_fig=True, filename="F3K3.png", title="Data with 3 features and 3 clusters")
    plt.show()


    plt.plot(range(0,115), bsom_3_features["all_NBME_avg_n4"])
    plt.show()

    """
    TODO: check for clusters ranging from k=2 to 10
    
    looks good for k =2, good clusters appear and from the Db score its verifies this fact
    as the clusters increases the db index
    
    smaller the index, better the cluster configuration
    
    """
    DB_indices_3_features=[]
    DB_scores_3_features=[]
    k_means_arr = []

    for k in range(2,11):
        k_means_arr.append(KMeans(n_clusters=k, n_iter=100, random_state=120))

    #cluster for each k
    preds = [kmeans.fit(bsom_3_features) for kmeans in k_means_arr]

    # for filename purpose
    fig_idx=2

    fd= open("centroids.txt", "w+")
    fd1 = open("membership.txt", "wb")

    # visualize the predictions
    for centroid, membership, labels in preds:
        visualize(centroid, membership, labels=bsom_3_features.columns, filename="fig_K_{}.png".format(fig_idx), save_fig=True, title="Visualization for K={}".format(fig_idx))
        davies_bouldin_index = DBIndex(membership)
        DB_indices_3_features.append(davies_bouldin_index.compute__DBIndex())
        fd.write("k= {} \ncentroid {} \n".format(fig_idx, centroid))
        if(fig_idx == 2):
            pkl.dump(membership, fd1)
        fig_idx += 1

        ## TODO: Question 2 - k =2 appears good due to clear distinction and far away centroids and less overlap
        #plt.savefig("fig_{}.png".format(i))

        DB_scores_3_features.append(davies_bouldin_score(bsom_3_features, np.array(labels["cluster_id"])))
    fd.close()
    fd1.close()
    print("^^^", DB_indices_3_features)
    plt.plot(range(2,11), DB_indices_3_features)
    #plt.plot(range(2, 11), DB_scores_3_features, "go")
    plt.title("DB index with 3 features")
    plt.xlabel("n_clusters")
    plt.ylabel("DB index")
    plt.savefig("Dbindex_3_feature.png", bbox_inches="tight")
    plt.show()


    ## Question 2 a.
    """
        adding a new feature "all_irats_avg_n34"
    """

    bsom_data = pd.read_csv("BSOM_DataSet_revised.csv")
    requried_cols = ["all_NBME_avg_n4", "all_PIs_avg_n131", "HD_final", "all_irats_avg_n34"]
    bsom_4_features= bsom_data[requried_cols]

    # store davies bouldin score for k - 2 to 11
    DB_indices_4_features = []
    DB_scores_4_features =[]
    preds = [kmeans.fit(bsom_4_features) for kmeans in k_means_arr]
    for centroid, membership, labels in preds:
        #TODO: pca plot
        #plt, ax = visualize(centroid, membership, labels=bsom_clean.columns)
        index = DBIndex(membership)
        DB_indices_4_features.append(index.compute__DBIndex())
        DB_scores_4_features.append(davies_bouldin_score(bsom_4_features, np.array(labels["cluster_id"])))
        ## TODO: Question 2 - k =2 appears good due to clear distinction and far away centroids and less overlap
        # i+=1
        #plt.savefig("fig_{}.png".format(i))
        #plt.show()

    """
     the DB index increases as the number of clusters increase.
     the lowest index value is 0.70, for k= 2 clusters

    """
    plt.plot(range(2, 11), DB_indices_3_features, label="DB indices with 3 features")
    plt.plot(range(2, 11), DB_indices_4_features, label="DB indices with 4 features")
    plt.title("DB index with 3 and 4 features")
    plt.xlabel("n_clusters")
    plt.ylabel("DB index")
    plt.legend()
    plt.show()

    # correlation plot
    corr = bsom_4_features.corr()
    sns.clustermap(corr)
    plt.title("correlation of features")
    plt.show()


    """
    Question 2b
    
    adding HA_Final to see the perf
    
    """
    bsom_data = pd.read_csv("BSOM_DataSet_revised.csv")
    requried_cols = ["all_NBME_avg_n4", "all_PIs_avg_n131", "HD_final", "all_irats_avg_n34", "HA_final"]
    bsom_5_features= normalize_columns(bsom_data[requried_cols])
    DB_indices_5_features = []
    DB_scores_5_features=[]
    # cluster for each k
    preds = [kmeans.fit(bsom_5_features) for kmeans in k_means_arr]

    # plot each config of K
    i = 1;
    for centroid, membership, labels in preds:
        # plt, ax = visualize(centroid, membership, labels=bsom_clean.columns )
        index = DBIndex(membership)
        DB_indices_5_features.append(index.compute__DBIndex())
        i += 1

        ## TODO: Question 2 - k =2 appears good due to clear distinction and far away centroids and less overlap
        # plt.savefig("fig_{}.png".format(i))

        DB_scores_5_features.append(davies_bouldin_score(bsom_5_features, np.array(labels["cluster_id"])))
        # plt.show()
    #print(DB_indices)
    # plt.plot(range(2, 11), DB_indices_5_features, label="5 features", marker="o")
    # plt.plot(range(2, 11), DB_indices_4_features, label="4 features",marker="*")
    # plt.title("DB index with 4 features vs 5 features")
    # plt.xlabel("n_clusters")
    # plt.ylabel("DB index")
    # plt.legend()
    #plt.show()


    ## plotting indices and scores
    for _idx,DB_indices in enumerate([DB_indices_3_features, DB_indices_4_features, DB_indices_5_features]):
        plt.plot(range(2,11), DB_indices, label="{} features".format(_idx+3), marker="*")
        print(_idx+3, DB_indices, "****")
    # for _idx, DB_scores in enumerate([DB_scores_3_features, DB_scores_4_features, DB_scores_5_features]):
    #     plt.plot(range(2, 11), DB_scores, label="{} features - using method".format(_idx + 3))
    plt.legend()
    plt.title("DB indicess with 5 features")
    plt.xlabel("n_clusters")
    plt.ylabel("DB index")
    #plt.show()


    #validation code
    DB_scores_3_features = []
    DB_scores_4_features = []
    DB_scores_5_features = []
    #k_means_arr = []
    k_means_arr_3_features=[]
    k_means_arr_4_features = []
    k_means_arr_5_features =[]
    for k in range(2,11):
        k_m=KMEANS(n_clusters=k, init="random",n_init=1, random_state=120)
        k_means_arr_3_features.append(k_m.fit_predict(bsom_3_features))
        k_means_arr_4_features.append(k_m.fit_predict(bsom_4_features))
        k_means_arr_5_features.append(k_m.fit_predict(bsom_5_features))
        print(k_m.cluster_centers_)

    for i in range(len(k_means_arr_3_features)):
        DB_scores_3_features.append(davies_bouldin_score(bsom_3_features, k_means_arr_3_features[i]))
        DB_scores_4_features.append(davies_bouldin_score(bsom_4_features, k_means_arr_4_features[i]))
        DB_scores_5_features.append(davies_bouldin_score(bsom_5_features, k_means_arr_5_features[i]))
        # index = DBIndex(membership_matrix)
        # print(index.compute__DBIndex())

    # plotting values from
    plt.plot(range(2,11), DB_scores_3_features, label=" 3 features from lib")
    plt.plot(range(2,11), DB_scores_4_features, label=" 4 features from lib")
    plt.plot(range(2,11), DB_scores_5_features, label=" 5 features from lib")
    # plt.plot(range(2,11), k_means_arr_3_features, label=" 3 features from lib")
    # plt.plot(range(2,11), k_means_arr_4_features, label=" 3 features from lib")
    # plt.plot(range(2,11), k_means_arr_5_features, label=" 3 features from lib")
    plt.xlabel("n_clusters")
    plt.ylabel("DB index")
    plt.legend()
    plt.show()



    # plt.plot(range(2, 11), DB_indices, label="5 features", marker="o")
    # plt.plot(range(2, 11), DB_indices_4_features, label="4 features", marker="*")
    # plt.plot(range(2,11), DB_scores_k_algo, label="k_means_algo", marker="^")
    # plt.title("DB index with 4 features vs 5 features")
    # plt.xlabel("n_clusters")
    # plt.ylabel("DB index")
    # plt.legend()
    # plt.show()
    #using synthetic data
    # test_data = pd.read_csv("Synthetic_test_data.csv")
    # test_data_ = test_data.drop("label", axis=1)
    # final_centroids, membership_matrix, _labels = KMeans(n_clusters=3, n_iter=100, random_state=120).fit(test_data_)
    # k_m = KMEANS(n_clusters=3, random_state=120 )
    # k_m.fit_predict(test_data_)
    # print(k_m.cluster_centers_)
    # print(final_centroids)
    #
    # print(np.equal(np.array(_labels["cluster_id"]).T, np.array(test_data["label"].T)))






















