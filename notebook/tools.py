import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import auc, roc_curve, recall_score, accuracy_score
from sklearn.metrics import precision_score, brier_score_loss, f1_score
import scipy
from scipy import stats
from scipy.stats import ks_2samp
from scipy.stats import median_absolute_deviation

plt.ion()
plt.show()
#import seaborn as sns
np.warnings.filterwarnings('ignore')


def get_cm_heatmap(cmn, title):

    fig, ax= plt.subplots(figsize=(6,5))
    sns.heatmap(cmn, annot=True, 
                ax = ax,
                linewidths=.4,
                #cbar_kws={'label': 'label_'},
                annot_kws={'size':16},
                fmt=".2f");
                            
    # labels, title and ticks
    ax.set_xlabel('Predicted', labelpad=20, fontsize=20);
    ax.set_ylabel('Actual', labelpad=20, fontsize=20); 
    ax.set_title(title, pad=0, size=20); 
    ax.xaxis.set_ticklabels(['0', '1'], size=20); 
    ax.yaxis.set_ticklabels(['0', '1'], size=20);
    
    
    
## Class to compute Metrics
class TemporalStability:
    """
    Class for estimating the temporal stability of a given ML model.
    Returns six different pieces of information:
    -----------------------------------------------------------
        1. MetricValues: standard metric values (i.e., auc, ks, recall,...) ('float')
        2. mean: average value ('float')
        3. variance: median_absolute_deviation ('float')
        4. slope: slope from a linear fitting ('float')
        5. periods: all periods involved ('string')

    Parameters by examples:
    -----------------------------------------------------------
        * df: pandas DataFrame
        * score_variable:'kiss_53'
        * target_variable: 'fpd_30'
        * temporal_variable: 'credit_analysis_created_at'
        * resampling_timescale: '1 M'
        * min_date_analysis: '2018-02-01'
        * max_date_analysis: '2019-07-01'
        * metric: 'auc' """

    def __init__(self,
                df,
                score_variable = 'kiss_53',
                target_variable = 'fpd_30',
                temporal_variable = 'credit_analysis_created_at',
                resampling_timescale = '1 M',
                min_date_analysis = None,
                max_date_analysis = None,
                metric = 'auc'):
       self.score_variable = score_variable
       self.target_variable = target_variable
       self.temporal_variable = temporal_variable
       self.resampling_timescale = resampling_timescale
       self.min_date_analysis = min_date_analysis
       self.max_date_analysis = max_date_analysis
       self.metric = metric
       self.metric_values = None
       self.n_periods = None
       self.dt = None
       self.df = df



    def MetricValues(self):
        """  """
        columns_to_read = [self.temporal_variable, self.target_variable, self.score_variable]
        df2 = self.df[columns_to_read].copy().dropna()

        def IV(y_pred, yreal, nbins):
            datos = pd.DataFrame({'score': y_pred, 'target': yreal})
            df_temp = pd.DataFrame()
            df_temp['scored'] = pd.qcut(datos['score'], nbins, labels=False, duplicates='drop')
            lst = []
            for i in range(df_temp['scored'].nunique()):
                val = list(df_temp['scored'].unique())[i]
                lst.append({'Good': df_temp[(df_temp['scored'] == val) & (datos['target'] == 0)].count()['scored'],
                            'Bad' : df_temp[(df_temp['scored'] == val) & (datos['target'] == 1)].count()['scored']})
            dset = pd.DataFrame(lst)
            dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
            dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
            dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
            dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
            dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
            iv_sum = dset['IV'].sum()
            # Removing temporal dataframes.
            dset.drop
            df_temp.drop
            datos.drop
            return np.round(iv_sum, 4)

       ### selecting periods.
        if self.min_date_analysis != None:
            mask = df2[self.temporal_variable] > self.min_date_analysis
            df2 = df2.loc[mask]

        if self.max_date_analysis != None:
            mask = df2[self.temporal_variable] <= self.max_date_analysis
            df2 = df2.loc[mask]

        df2[self.temporal_variable] = pd.to_datetime(df2[self.temporal_variable].to_list())
        # df2.fpd_30 = df2.fpd_30.astype(float)
        df2 = df2.set_index(self.temporal_variable)

        self.dt = df2.resample(self.resampling_timescale).count().index
        self.n_periods = self.dt.shape[0]

        ### computing values according to selected metric
        if self.metric == 'all':
            self.metric_values = np.zeros((self.n_periods, 8), dtype=float)
        else:
            self.metric_values = np.zeros(self.n_periods)
        for ttt in range(self.n_periods):
            if ttt<1: cond = (df2.index < self.dt[1])
            elif ttt == self.n_periods-1: cond = (df2.index > self.dt[-2])
            else: cond = (df2.index >= self.dt[ttt-1]) & (df2.index < self.dt[ttt])

            yreal  = df2.loc[cond, self.target_variable].values.astype(int)
            y_pred = df2.loc[cond, self.score_variable].values

            if len(yreal)>0:

                if self.metric == 'auc':
                    fpr, tpr, th = roc_curve(yreal, y_pred)
                    try: 
                        self.metric_values[ttt] = np.round(auc(fpr, tpr),4)
                    except: 
                        print('pepe')
                        pass

                elif self.metric == 'recall':
                    try: self.metric_values[ttt] = np.round(recall_score(yreal, y_pred.round(), average=None)[0],4)
                    except: pass

                elif self.metric == 'precision':
                    try: self.metric_values[ttt] = np.round(precision_score(yreal, y_pred.round(), average=None)[0],4)
                    except: pass

                elif self.metric == 'accuracy':
                    try: self.metric_values[ttt] = np.round(accuracy_score(yreal, y_pred.round()),4)
                    except: pass

                elif self.metric == 'brier_score_loss':
                    try: self.metric_values[ttt] = np.round(brier_score_loss(yreal, y_pred.round()),4)
                    except: pass

                elif self.metric == 'f1_score':
                    try: self.metric_values[ttt] = np.round(f1_score(yreal, y_pred.round(), average=None)[0],4)
                    except: pass

                elif self.metric == 'ks':
                    try: self.metric_values[ttt] = np.round(ks_2samp(y_pred[yreal == 0], y_pred[yreal == 1]).statistic,4)
                    except: pass
                elif self.metric == 'iv':
                    try: self.metric_values[ttt] = IV(y_pred, yreal, 10)
                    except: pass
                elif self.metric == 'all':
                    try:
                        #auc
                        fpr, tpr, th = roc_curve(yreal, y_pred)
                        self.metric_values[ttt, 0] = np.round(auc(fpr, tpr),4)
                        #recall
                        self.metric_values[ttt, 1] = np.round(recall_score(yreal, y_pred.round(), average=None)[0],4)
                        #precision
                        self.metric_values[ttt, 2] = np.round(precision_score(yreal, y_pred.round(), average=None)[0],4)
                        #accuracy
                        self.metric_values[ttt, 3] = np.round(accuracy_score(yreal, y_pred.round()),4)
                        #brier_score_loss
                        self.metric_values[ttt, 4] = np.round(brier_score_loss(yreal, y_pred.round()),4)
                        #f1_score
                        self.metric_values[ttt, 5] = np.round(f1_score(yreal, y_pred.round(), average=None)[0],4)
                        #ks
                        self.metric_values[ttt, 6] = np.round(ks_2samp(y_pred[yreal == 0], y_pred[yreal == 1]).statistic,4)
                        #iv
                        self.metric_values[ttt, 7] = IV(y_pred, yreal, 10)
                    except: pass
            else:
                pass #print('End of report!')

        if self.metric == 'all':
            self.metricas = ['auc', 'recall', 'precision', 'accuracy', 'brier_score_loss', 'f1_score', 'ks', 'IV']
            self.metrics_frame = pd.DataFrame(data=self.metric_values, columns=self.metricas, index=self.dt)
        else:
            self.metrics_frame = pd.DataFrame(data={self.metric:self.metric_values}, index=self.dt)
        self.metrics_frame.index.name = 'time intervals'
        return(self.metrics_frame)


    def mean(self):
        """  """
        return(np.round(np.mean(self.metric_values),3))


    def variance(self):
        """  """
        return(np.round(median_absolute_deviation(self.metric_values),3))


    def slope(self):
        """  """
        def plot_linear_fitting(y):
            nx = len(y)
            x = np.arange(nx)
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            return z, p

        metric_values_redu = self.metric_values[self.metric_values>0]
        z, p = plot_linear_fitting(metric_values_redu)
        new_base = np.linspace(0.,self.n_periods, 10)
        slope = plot_linear_fitting(metric_values_redu)[0][0] 
        return(np.round(slope,3))


    def periods(self):
        return(self.dt.strftime('%Y-%m-%d').to_list()) 
    
    def n_periods(self):
        return self.n_periods
    
    def dts(self):
        return self.dt
    
    def metric_values(self):
        return self.metric_values  
    
    
    
    
    
    
def Plot(n_periods, metric_values, featurename, metricname, contours, window, DepPer):
        """  """
        if window != '1': plt.figure(window, figsize=(15, 5))
        else:  plt.figure(1, figsize=(15, 5))
        plt.clf()
        plt.plot(n_periods, metric_values, '-o', lw=8, ms=15)
        plt.plot(n_periods, metric_values, 'o', ms=8, color='white')
        xmin, xmax = plt.xlim()
        plt.xticks(n_periods, n_periods.strftime('%Y-%m-%d').to_list(), rotation=25, fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        plt.title(featurename, size=20, weight = 'bold')
        plt.xlabel('Timeline', size=18, labelpad=15)
        plt.ylabel('%s'%(metricname), size=18, labelpad=15)
        label_info = '$\mu$ = %.2f''\n''$\sigma$ = %.2f'%(metric_values.mean(), metric_values.std())
        if contours:
            dev = n_periods<= DepPer #'2020-01-01'
            plt.axvspan(n_periods[dev].min(), n_periods[dev].max(), 0, 1., color='tab:green', alpha=0.2)
            plt.axvspan(n_periods[~dev].min(), n_periods[~dev].max(), 0, 1., color='tab:orange', alpha=0.2)
        plt.legend([label_info], loc='best', fontsize=22)    



def plot_maxks_score(df, scorename, target, bins, plots=True):
    
    # Making sure there are not NaN values.
    df = df.query(target + " == " + target)
    
    # Reading necessary data
    df_0 = df.query(target + " == 0").dropna()
    df_1 = df.query(target + " == 1").dropna()
    
    # Setting parameters
    minv  = df[scorename].dropna().values.min()
    maxv  = df[scorename].dropna().values.max()
    nbins = bins
    base  = np.linspace(minv, maxv, nbins)
    base2 = base[:-1] + ((base[1]-base[0])/2.)
        
    # Retrieving info from bins
    a1, a2 = np.histogram(df_0[scorename].values, base)
    b1, b2 = np.histogram(df_1[scorename].values, base)
    a1c   = np.cumsum(a1/a1.sum())
    b1c   = np.cumsum(b1/b1.sum())
    
    delta = abs(a1c-b1c)
    pos   = np.argmax(delta)
    print("-------------------------------")
    print('score_bin_at_KSmax =',pos+1)
    print('score_value_at_KSmax =', np.round(base2[pos],3))
    print('KS_max =', np.round(delta[pos],3))
    print('Values at KSmax = %.2f, %.2f'%(np.round(a1c[pos],2), np.round(b1c[pos],2)))
    print("-------------------------------")
    
    if plots:
        plt.figure(1, figsize=(15, 6))
        plt.clf()

        plt.subplot(121)
        a1, a2, a3 = plt.hist(df_0[scorename].values, base, alpha=0.2)
        b1, b2, b3 = plt.hist(df_1[scorename].values, base, alpha=0.2)
        plt.legend([target+' = 0', target+' = 1'], loc='best', fontsize=16, facecolor='white')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        plt.xlabel('score', size=20)
        plt.ylabel('distribution', size=20)

        plt.subplot(122)
        plt.plot(base2, a1c, '-o', lw=4, ms=10)
        plt.plot(base2, b1c, '-o', lw=4, ms=10)
        plt.vlines(base2[pos], a1c[pos], b1c[pos], linestyles='solid', lw=6, color='grey', alpha=0.75)
        plt.xlabel('score', size=20)
        plt.ylabel('cumulative distribution', size=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend([target+' = 0', target+' = 1', 'KS$\_{max}$=%.2f'%(delta[pos])], loc='best', fontsize=16, facecolor='white')
        plt.grid()
        #plt.savefig(local_path + 'ks_%s.png'%(scorename), dpi=100)        