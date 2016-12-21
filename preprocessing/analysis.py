import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif


def apply_pca(features, n_components):
    pca = PCA(n_components = n_components)
    pca.fit(features)
    reduced_features = pca.transform(features)
    print 'PCA variance ratios:', pca.explained_variance_ratio_
    print 'PCA sum of variance ratios:', sum(pca.explained_variance_ratio_)
    print 'PCA noise variance:', pca.noise_variance_
    pickle.dump(reduced_features, open("features_pca1.p", "wb"))

    plt.clf()
    x = range(1, len(pca.explained_variance_ratio_)+1)
    plt.plot(x, pca.explained_variance_ratio_, marker='o')
    plt.yscale('log')
    plt.ylim([0.00000000000000000000000000000000001, 10])
    plt.ylabel('Variance ratio')
    plt.xlabel('Component')
    plt.title('Component variances')

    plt.show()

    plt.clf()
    #x = range(1, len(pca.explained_variance_ratio) + 1)
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    for i,t in enumerate(reduced_features):
        if labels[i] == 1:
            x1.append(t[0])
            y1.append(t[1])
        else:
            x2.append(t[0])
            y2.append(t[1])

    plt.scatter(x2,y2,marker='.',color='b',alpha=0.66,label='failure')
    plt.scatter(x1,y1,marker='.',color='r',alpha=0.33,label='success')

    plt.legend(loc=4)


    plt.ylabel('Component 2')
    plt.xlabel('Component 1')
    plt.xlim([-1200,1000])
    plt.title('Projection of first two components')


def apply_kernel_pca(features, labels):
    plt.clf()
    #x = range(1, len(pca.explained_variance_ratio) + 1)
    x1=[]
    y1=[]
    x2=[]
    y2=[]

    kernel_pca = KernelPCA(n_components=2, kernel='sigmoid')
    reduced_features = kernel_pca.fit_transform(features, labels)
    for i, t in enumerate(reduced_features):
        if labels[i] == 1:
            x1.append(t[0])
            y1.append(t[1])
        else:
            x2.append(t[0])
            y2.append(t[1])

    plt.scatter(x2, y2, marker='.', color='b', alpha=0.66, label='failure')
    plt.scatter(x1, y1, marker='.', color='r', alpha=0.33, label='success')
    plt.legend(loc=4)
    plt.ylabel('Component 2')
    plt.xlabel('Component 1')
    plt.xlim([-1200, 1000])
    plt.title('Projection of first two components')
    plt.show()


def apply_anova_f_value_test(features, labels, encoder):
    (f_val,_) = f_classif(features, labels)
    sort_scores = [i[0] for i in sorted(enumerate(f_val), key=lambda x:x[1], reverse=True)]
    for i in sort_scores:
        print encoder.feature_names_[i], ':', f_val[i]


def apply_variance_threshold_selection(features, labels, encoder):
    sp = VarianceThreshold()
    sp.fit(features,labels)
    print sp.scores_

    for i in range(len(encoder.feature_names_)):
        print encoder.feature_names_[i], ':', sp.variances_[i]


def plot_progress_measure():
    x = np.linspace(0, 15, 75)
    y1 = []
    y2 = []
    y3 = []

    for i in x:
        if i < 10.0:
            y3.append(0.0)
            y1.append(i/10.0)
            y2.append((i/10.0)**2)
        else:
            y3.append(1.0)
            y1.append(1.0)
            y2.append(1.0)

    plt.clf()
    plt.plot(x, y1, label='1st down')
    plt.plot(x, y2, label='2nd down')
    plt.plot(x, y3, label='3rd/4th down')
    plt.ylim([0, 1.1])
    plt.xlim([0, 13])
    plt.legend(loc=2)
    plt.xticks([0, 5, 10], ['0', 'togo/2', 'togo'])
    plt.ylabel('Progress score')
    plt.xlabel('Distance')
    plt.title('Progress label')
    plt.show()

    # second plot:
    y1 = []
    y2 = []
    y3 = []

    for i in x:
        if i < 10.0:
            y3.append(0.0)
            y1.append((i*2)/10.0)
            y2.append(i/10.0)
        else:
            y3.append(1 + float(i - 10.0) / 10.0)
            y1.append(1 + float(i - 10.0) / 10.0)
            y2.append(1 + float(i - 10.0) / 10.0)

    plt.clf()
    plt.plot(x, y1, label='1st down')
    plt.plot(x, y2, label='2nd down')
    plt.plot(x, y3, label='3rd/4th down')
    plt.ylim([0, 2.2])
    plt.xlim([0, 15])
    plt.legend(loc=2)
    plt.xticks([0, 5, 10], ['0', 'togo/2', 'togo'])
    plt.ylabel('Progress score')
    plt.xlabel('Distance')
    plt.title('Progress label (Version 1)')
    plt.show()