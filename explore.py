import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statistics


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
    plt.axvline(6000, label = 'Average lotsize', ls='--', color='red')
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

    print(p)