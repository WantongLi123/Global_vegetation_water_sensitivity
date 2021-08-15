#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import cmaps
cmap_color=cmaps.ncl_default          #如需反转colorbar，其方法是cmap_color=cmap_color.reversed()
import imageio
import pymannkendall as mk
from matplotlib import rc

variable = [r'$\frac{\partial{ELAI}}{\partial{P}}$',r'$\frac{\partial{ELAI}}{\partial{SM}}$',r'$\frac{\partial{ELAI}}{\partial{T}}$',r'$\frac{\partial{ELAI}}{\partial{SW}}$',r'$\frac{\partial{ELAI}}{\partial{VPD}}$']
unit = ['$\mathregular{m^{-1}}$', '$\mathregular{m^3/m^3}$', '$\mathregular{˚C^{-1}}$', '$\mathregular{MJ^{-1}}$', '$\mathregular{KPa^{-1}}$']
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
plt.rcParams.update({'font.size': 10})
plt.rcParams['axes.unicode_minus'] = False # correct the sign of figure'' labels when there is a negative value

def data_path(filename):
    file_path = "{path}/{filename}".format(
        path="/Net/Groups/BGI/scratch/wantong",
        filename=filename
    )
    return file_path

def read_data(path):
    data = np.load(path)
    print(log_string, path, 'Read data')
    return data

if __name__ == '__main__':
    log_string = 'heatmap :'


    # plot 3-y sensitivity
    fig = plt.figure(figsize=(2.2, 2), dpi=300, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(1982, 2018, 1)
    x1 = np.arange(1982, 2018, 0.5)
    y1 = [0.008,0.008,0.008,0.008,0.008,np.nan,0.01,0.01,0.01,0.01,0.01,np.nan,0.013,0.013,0.013,0.013,0.013,np.nan,
          0.011,0.011,0.011,0.011,0.011,np.nan,0.015,0.015,0.015,0.015,0.015,np.nan,0.013,0.013,0.013,0.013,0.013,np.nan,
          0.014,0.014,0.014,0.014,0.014,np.nan,0.017,0.017,0.017,0.017,0.017,np.nan,0.019,0.019,0.019,0.019,0.019,np.nan,
          0.016,0.016,0.016,0.016,0.016,np.nan,0.019,0.019,0.019,0.019,0.019,np.nan,0.02,0.02,0.02,0.02,0.02,np.nan]

    y2 = np.linspace(0.008, 0.02, 36)
    y3 = [0.014]*36
    ax.plot(x1[0:6], y1[0:6], '-', color='yellow',alpha=1, label='3-Y-block sensitivity')
    ax.plot(x1[6:12], y1[6:12], '-', color='yellow',alpha=0.9, label='3-Y-block sensitivity')
    ax.plot(x1[12:18], y1[12:18], '-', color='yellow', alpha=0.8, label='3-Y-block sensitivity')
    ax.plot(x1[18:24], y1[18:24], '-', color='yellow', alpha=0.7, label='3-Y-block sensitivity')
    ax.plot(x1[24:30], y1[24:30], '-', color='yellow', alpha=0.6, label='3-Y-block sensitivity')
    ax.plot(x1[30:36], y1[30:36], '-', color='yellow', alpha=0.5, label='3-Y-block sensitivity')
    ax.plot(x1[36:42], y1[36:42], '-', color='green', alpha=0.5, label='3-Y-block sensitivity')
    ax.plot(x1[42:48], y1[42:48], '-', color='green', alpha=0.6, label='3-Y-block sensitivity')
    ax.plot(x1[48:54], y1[48:54], '-', color='green', alpha=0.7, label='3-Y-block sensitivity')
    ax.plot(x1[54:60], y1[54:60], '-', color='green', alpha=0.8, label='3-Y-block sensitivity')
    ax.plot(x1[60:66], y1[60:66], '-', color='green', alpha=0.9, label='3-Y-block sensitivity')
    ax.plot(x1[66:72], y1[66:72], '-', color='green', alpha=1, label='3-Y-block sensitivity')

    # for pic, v in zip(range(12), [2,8,12,18,24,30,36,42,48,54,60,66]):
    #     print(v,x1[v], y1[v],cmap[pic])
    #     ax.scatter(x1[v], y1[v], 'x', c=plt.get_cmap('YlGn'))

    ax.plot([1982,2017], [0.008,0.02], '-', color='blue', alpha=0.1, linewidth=5, label='Trends of sensitivity')
    # ax.plot(x, y2, '--', color='blue')
    # ax.plot(x, y3, '-', color='red', label='Overall sensitivity')
    # ax.legend(frameon=False)

    ax.set_xticks([1982, 2000, 2017])
    ax.set_yticks([0.008, 0.014, 0.02])
    ax.set_xlabel('Year')
    ax.set_ylabel('LAI sensitivity to SMsurf')

    # # plot shape dependence
    # fig = plt.figure(figsize=(6, 4), dpi=300, tight_layout=True)
    # ax = fig.add_subplot(1, 1, 1)
    # x = np.random.uniform(low=-10, high=10, size=50)
    # y1=x*0.006
    # y1 = y1 + np.random.normal(0, 0.02, y1.shape)
    # y2 = x * 0.009
    # y2 = y2 + np.random.normal(0, 0.02, y1.shape)
    # y3 = x * 0.012
    # y3 = y3 + np.random.normal(0, 0.02, y1.shape)
    # y4 = x * 0.015
    # y4 = y4 + np.random.normal(0, 0.02, y1.shape)
    #
    # ax.scatter(x,y1,color='yellow', marker='x', s=20, label='1982-1984')
    # ax.scatter(x,y2,color='yellow',alpha=0.4,marker='x', s=20, label='1985-1987')
    # ax.scatter(x,y3,color='green',alpha=0.4, marker='x', s=20, label='2012-2014')
    # ax.scatter(x,y4,color='green',marker='x', s=20, label='2015-2017')
    # ax.plot([-10,10],[-0.05,0.04],color='yellow', linewidth=0.5)
    # ax.plot([-10,10],[-0.1,0.1],color='yellow',alpha=0.4, linewidth=0.5)
    # ax.plot([-10,10],[-0.12,0.14],color='green',alpha=0.4, linewidth=0.5)
    # ax.plot([-10,10],[-0.16,0.16],color='green', linewidth=0.5)
    #
    # # x_all = np.random.uniform(low=-10, high=10, size=200)
    # # y_all = x_all * 0.014
    # # y_all = y_all + np.random.normal(0, 0.05, y_all.shape)
    # # ax.scatter(x_all,y_all,color='red',marker='x',s=10, label='1982-2017')
    # # ax.plot([-10, 10], [-0.14, 0.14], color='red', linewidth=0.5)
    #
    # ax.legend(frameon=False)
    # ax.set_ylim(-0.2,0.2)
    # ax.set_xticks([-10,-5,0,5,10])
    # ax.set_yticks([-0.2,-0.1,0,0.1,0.2])
    # ax.set_xlabel('SMsurf anomaly')
    # ax.set_ylabel('Contributions on LAI\n(Shap values)')

    fig.savefig(data_path('study2/results/result_april/figure1/conceptual_figure.jpg'),bbox_inches='tight')

    print('end')