import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
import scipy.cluster.hierarchy
import scipy.spatial.distance

features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']
f_len = len(features)

data = pd.read_excel(r'C:\Users\Kirill Voronovich\Desktop\Data.xlsx', index_col='Регион').loc[:, features]
print(data.head())


#ss = StandardScaler()
#scaled_data = pd.DataFrame(ss.fit_transform(data), columns=data.columns, index=data.index)
#print(scaled_data)

methods = ['complete', 'ward', 'single', 'average', 'weighted']
for method in methods:
     Z = hierarchy.linkage(data, method=method, optimal_ordering=True)
     #строим дендрограмму

     plt.figure(figsize=(10,6)) # задаем размеры окна с графиками

     hierarchy.dendrogram(Z, labels=data.index, leaf_font_size=10)

#  #Z - результат кластеризации, labels - названия строк (ось Х), leaf_font_size - размер шрифта

     plt.title('{} method'.format(method))
     plt.show()

CLUSTER_METHODS = ["complete", "ward", "kmeans", "single", "average", 'weighted']
N_CLUSTERS = {
    "complete": 7,
    "ward": 5,
    "kmeans": 5,
    "single": 5,
    "average": 7,
    "weighted": 5,
}
# метод полных связей
complete = AgglomerativeClustering(n_clusters=N_CLUSTERS['complete'], linkage='complete')
complete.fit(data)

# метод варда
ward = AgglomerativeClustering(n_clusters=N_CLUSTERS['ward'], linkage='ward')
ward.fit(data)
# метод kmeans
kmeans = KMeans(n_clusters=N_CLUSTERS['kmeans'], random_state=36)
kmeans.fit(data)
# метод одиночных связей
single = AgglomerativeClustering(n_clusters=N_CLUSTERS['single'], linkage='single')
single.fit(data)

# метод невзвешенного попарного среднего
average = AgglomerativeClustering(n_clusters=N_CLUSTERS['average'], linkage='average')
average.fit(data)
# метод взвешенного попарного среднего
Z = scipy.cluster.hierarchy.weighted(scipy.spatial.distance.pdist(data))
weighted = scipy.cluster.hierarchy.fcluster(Z, 4.84, criterion='distance')

data['complete'] = complete.labels_
data['ward'] = ward.labels_
data['kmeans'] = kmeans.labels_
data['single'] = single.labels_
data['average'] = average.labels_
data['weighted'] = weighted


def mean_df(method, n_clust):
    mean_data = np.array([]).reshape(0, f_len + 1)
    for n in range(n_clust):
        tmp = []
        for j in range(f_len):
            tmp.append(data[data[method] == n].iloc[:, j].mean())

        tmp.append(data[data[method] == n].shape[0])
        mean_data = np.vstack((mean_data, np.array(tmp).reshape(1, f_len + 1)))

    return mean_data


columns = features + ['count']
means = {}
dfs = []

for method, n in N_CLUSTERS.items():
    means[method] = pd.DataFrame(
        mean_df(method, n),
        columns=columns,
        index=["{}_{}".format(method, i) for i in range(n)]
    )


for method in CLUSTER_METHODS:
    print(means[method])

for method in CLUSTER_METHODS:
    # в cur_mean записывается элемент словаря means, соот-ветствующий ключу method (т.е. средние значения признаков в кластерых для метода method)
    cur_mean = means[method]
    plt.figure(figsize=(6,4))
   #для каждого кластера из данного метода строим график
    for n in range(cur_mean.shape[0]):
        plt.plot(features, cur_mean.iloc[n, :-1].values, marker='o', label='cluster {}'.format(n))
        plt.legend(loc = 'upper left') #легенда графика
    plt.title('{} method'.format(cur_mean.index[0][:-2])) #назва-ние
    plt.show()

features = ['X' + str(i) for i in range(1, 9)]
cluster_dict = {}

for method, n in N_CLUSTERS.items():
    # выводим заголовки методов
    print('{} method\n'.format(method))
    dfs = []
    # для каждого кластера проходимся по объектам и запи-сываем их в словарь
    for i in range(n + 1):
        cluster_dict['{}_{}'.format(method, i)] = data[data[method] == i][features]

        # выводим на экран состав кластеров
        print('Состав кластера {}: {}'
              .format(i, cluster_dict['{}_{}'.format(method, i)].index.values))
    print('\n')

mean_data = pd.DataFrame()
for method in CLUSTER_METHODS:
    mean_data = mean_data.append(means[method])
print(mean_data)

Dw=pd.DataFrame(columns=features)
quality=pd.DataFrame(columns=['Q1','method'])
for j in CLUSTER_METHODS:
    for i in data[j].unique():
        Dw=Dw.append(((data[data[j]==i]-
            pd.pivot_table(data,columns=j)[4:].loc[:,i])**2).sum(),ignore_index=True)
    print(f'Q1={np.sum(Dw.to_numpy())}, method: {j}',)
    quality=quality.append({'Q1':np.sum(Dw.to_numpy()),'method':j},ignore_index=True)
    Dw=pd.DataFrame(columns=features)
print(quality.T)