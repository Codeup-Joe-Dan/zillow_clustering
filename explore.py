import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statistics
from sklearn.cluster import KMeans



def target_dist(df):
    '''
    Creates a hist plot to show the distribution of log_error for properties 
    Accepts a dataframe and creates/displays a plot, along with min/max/mean/stddev
    '''
    # Plot Distribution of target variable
    plt.figure(figsize=(24,12))
    sns.set(font_scale=2)
    plt.title('Distribution of Log Error')
    sns.histplot(data=df, x='logerror')
    plt.xlim(-1, 1)

    plt.grid(False)
    plt.axvline(df.logerror.mean(), color='k', linestyle='dashed', linewidth=.5)
    min_ylim, max_ylim = plt.ylim()
    plt.text(df.logerror.mean()*1.5, max_ylim*0.9, 'Mean: {:,.4f}'.format(df.logerror.mean()))
    plt.text(df.logerror.mean()-.85, max_ylim*0.9, 'Minimum: {:,.4f}'.format(df.logerror.min()))
    plt.text(df.logerror.mean()-.85, max_ylim*0.85, 'Maximum: {:,.4f}'.format(df.logerror.max()))
    plt.text(df.logerror.mean()-.85, max_ylim*0.80, 'Std Dev: {:,.4f}'.format(statistics.stdev(df.logerror)))

    plt.show()

def county_plot(df):
    '''
    Function to display a factor plot with the average logerror of properties in each county, 
    and perform Analysis of Variance test for mean logerror in the 3 counties
    Accpets a dataframe, creates a plot, prints mean values by county and whether the null hypothesis is confirmed
    '''
    # create subsets of df for each county
    orange = df[df.county == 'Orange']
    ventura = df[df.county == 'Ventura']
    la = df[df.county == 'Los Angeles']

    # get baseline logerror rate for the horizontal line
    baseline = df.abserror.mean()

    # display factorplot
    p = sns.factorplot( x="county", y="abserror",  data=df, size=5, 
                   aspect=2, kind="bar", palette="bright", ci=None,
                   edgecolor=".2")
    plt.grid(False)
    plt.axhline(baseline, label = 'Overall average logerror', ls='--')
    p.set_ylabels("Average Log Error")
    p.set_xlabels("County")
    plt.title('Does County affect Log Error?')
    plt.show()
    # output values in each county
    print('Average logerror of those in Los Angeles County is ', "{:,.2f}".format((la.abserror.mean())))
    print('Average logerror value of those in Ventura County is', "{:,.2f}".format((ventura.abserror.mean())))
    print('Average logerror value of those in Orange County is', "{:,.2f}".format((orange.abserror.mean())))
    print("")
    print("")
    
    # perform ANOVA test and display results
    alpha = .05
    f, p = stats.f_oneway(orange.abserror, ventura.abserror, la.abserror) 
    if p < alpha:
        print("We reject the Null Hypothesis")
    else:
        print("We confirm the Null Hypothesis")

def lotsize_plot(df):
    '''
    Function to display a rel plot with the average logerror of properties by lot size, 
    and perform T-test for logerror of small properties
    Accpets a dataframe, creates a plot, prints mean values and whether the null hypothesis is confirmed
    '''

    # create subsets of df for large and small lotsize
    small = df[df.lotsize < 6000]
    
    # display plot
    p = sns.relplot( x="lotsize", y="abserror",  data=df, size=5, 
                   aspect=2, palette="bright", ci=None,
                   edgecolor=".2")
    plt.grid(False)
    # plt.axvline(6000, label = 'Average lotsize', ls='--', color='red')
    plt.axvspan(-5, 6000, color='green', alpha=0.1)

    p.set_ylabels("Log Error")
    p.set_xlabels("Lot Size")
    plt.title('Does Lot Size impact Log Error?')
    plt.show()
    # output values 
    print('Average logerror of small properties ', "{:,.4f}".format((small.abserror.mean())))
    print('Average logerror of all properties ', "{:,.4f}".format((df.abserror.mean())))
    print("")
    print("")

    # perform T-Test test and display results
    alpha = .05

    # perform test
    t, p = stats.ttest_1samp(small.abserror, df.abserror.mean())
    if p < alpha:
        print("We reject the Null Hypothesis")
    else:
        print("We confirm the Null Hypothesis")

def find_k(df, cluster_vars):
    sse = []
    k_range = range(2,20)
    for k in k_range:
        kmeans = KMeans(n_clusters=k)

        kmeans.fit(df[cluster_vars])

        # inertia: Sum of squared distances of samples to their closest cluster center.
        sse.append(kmeans.inertia_) 

    # create a dataframe with all of our metrics to compare them across values of k: SSE, delta, pct_delta
    k_comparisons_df = pd.DataFrame(dict(k=k_range[0:-1], 
                             sse=sse[0:-1]
                             ))

    # plot k with inertia
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 15})
    plt.plot(k_comparisons_df.k, k_comparisons_df.sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title(f'The Elbow Method to find the optimal k\n for {cluster_vars}\nFor which k values do we see large decreases in SSE?')
    # plt.annotate('Elbow', xy=(5, k_comparisons_df.sse[4]),  
    #         xytext=(5, 5), textcoords='axes fraction',
    #         arrowprops=dict(facecolor='red', shrink=0.05),
    #         horizontalalignment='right', verticalalignment='top',
            # )
    #plt.ticklabel_format(style='plain')
    plt.yticks([])
    plt.show()

    return 

def create_clusters(df, k, cluster_vars):
    # create kmean object
    kmeans = KMeans(n_clusters=k, random_state = 123)

    # fit to train and assign cluster ids to observations
    kmeans.fit(df[cluster_vars])

    return kmeans


def get_centroids(kmeans, cluster_vars, cluster_name):
    # get the centroids for each distinct cluster...

    centroid_col_names = ['centroid_' + i for i in cluster_vars]

    centroid_df = pd.DataFrame(kmeans.cluster_centers_, 
                               columns=centroid_col_names).reset_index().rename(columns={'index': cluster_name})

    return centroid_df

def cluster_plot(df):
    '''
    creates catplot for clusters and performs ANOVA test to determine significance
    accepts a dataframe and cluster variables, plots log error by cluster, and prints test result
    '''
    
    sns.catplot(x ="cluster",
             y ="logerror",
             data = df, size=5)
    plt.title("Log Error by cluster")
    plt.show()

    cluster1 = df[df.cluster == 0]
    cluster2 = df[df.cluster == 1]
    cluster3 = df[df.cluster == 2]
    cluster4 = df[df.cluster == 3]
    cluster5 = df[df.cluster == 4]

    print('H0:  The logerrors are not significantly different')
    print('Ha:  The logerrors are significantly different')

    if len(cluster5) == 0:
        alpha = .05
        f, p = stats.f_oneway(cluster1.logerror, cluster2.logerror, cluster3.logerror, cluster4.logerror) 
        if p < alpha:
            print("We reject the Null Hypothesis")
        else:
            print("We confirm the Null Hypothesis")
            print(p)
    else:
        alpha = .05
        f, p = stats.f_oneway(cluster1.logerror, cluster2.logerror, cluster3.logerror, cluster4.logerror, cluster5.logerror) 
        if p < alpha:
            print("We reject the Null Hypothesis")
        else:
            print("We confirm the Null Hypothesis")
            print(p)

def add_clusters(kmeans, train, validate, test, cluster_vars):
    '''
    function to create clusters and add one-hot encoded named columns to train,
    validate, and test dataframes.
    Accepts the fit kmeans, three dataframes, and a list of cluster variables
    returns the three dataframes with named and one-hot encoded columns
    '''

    # add clusters to original train/validate/test
    train['cluster'] = kmeans.predict(train[cluster_vars])
    validate['cluster'] = kmeans.predict(validate[cluster_vars])
    test['cluster'] = kmeans.predict(test[cluster_vars])

    # rename clusters
    train['cluster'] = train.cluster.map({0 : 'littlelot_mediumprice',
                                          1 : 'littlelot_highprice',
                                          2 : 'largelot_mediumprice',
                                          3 : 'littlelot_lowprice'})

    validate['cluster'] = validate.cluster.map({0 : 'littlelot_mediumprice',
                                                1 : 'littlelot_highprice',
                                                2 : 'largelot_mediumprice',
                                                3 : 'littlelot_lowprice'})

    test['cluster'] = test.cluster.map({0 : 'littlelot_mediumprice',
                                        1 : 'littlelot_highprice',
                                        2 : 'largelot_mediumprice',
                                        3 : 'littlelot_lowprice'})

    # One-Hot-Encode
    dummies = pd.get_dummies(train['cluster'],drop_first=False)
    train = pd.concat([train, dummies], axis=1)
    train.drop(columns='cluster', inplace=True)

    dummies = pd.get_dummies(validate['cluster'],drop_first=False)
    validate = pd.concat([validate, dummies], axis=1)
    validate.drop(columns='cluster', inplace=True)

    dummies = pd.get_dummies(test['cluster'],drop_first=False)
    test = pd.concat([test, dummies], axis=1)
    test.drop(columns='cluster', inplace=True)
    
    return train, validate, test



