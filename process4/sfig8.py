#!/usr/bin/env python
import sys
import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from matplotlib import rc
import matplotlib.gridspec as gridspec
import pymannkendall as mk
import math

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
plt.rcParams.update({'font.size': 6})
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['lines.linewidth'] = 1

tit1 = [r'$\frac{\partial{LAI}}{\partial{SMsurf}}$',r'$\frac{\partial{LAI}}{\partial{SMroot}}$']
tit2 = [r'$\frac{\partial{NDVI}}{\partial{SMsurf}}$',r'$\frac{\partial{NDVI}}{\partial{SMroot}}$']
unit1 = ' ($\mathregular{mm^{-1}}$ per 3 years)'
unit2 = ' ($\mathregular{mm^{-1}}$)'

def data_path(filename):
    file_path = "{path}/{filename}".format(
        path="/Net/Groups/BGI/scratch/wantong",
        filename=filename
    )
    return file_path

def read_data(path):
    data = np.load(path)
    print(log_string, path, 'read data')
    return data

def temporal_vari_sen(Sensi_3Y, Sensi_all):
    year = np.shape(Sensi_3Y[2, :, :, :, :])[0]
    temporal_vari_sensi = np.zeros((2, year, 360, 720)) * np.nan
    for v, index in zip([1, 2], [0, 1]):
        sensiSlope_3Y = Sensi_3Y[2, :, v, :, :]
        sensiSlope_all = Sensi_all[3, v, :, :]
        sensiPvalue_all = Sensi_all[4, v, :, :]
        sensiSlope_all = np.repeat(sensiSlope_all[np.newaxis, :, :], year, axis=0)
        sensiPvalue_all = np.repeat(sensiPvalue_all[np.newaxis, :, :], year, axis=0)
        sensiSlope_3Y[sensiSlope_all <= 0] = np.nan
        sensiSlope_3Y[sensiPvalue_all >= 0.01] = np.nan
        temporal_vari_sensi[index, :, :, :] = sensiSlope_3Y
    return(temporal_vari_sensi)

def plot_with_interquartile(ax, ELAI, NDVI3g, LAImodis, KNDVImodis):
    #draw 5 lines
    year = [[1982,2018],[2000,2018],[1982,2015],[2000,2018]]
    color = ['blue','steelblue', 'lightblue', 'darkblue']
    label = ['EnLAI', 'LAImodis', 'NDVI3g', 'KNDVImodis']
    for v, name in zip(range(2), [ELAI, LAImodis]):
        if v==0:
            print(np.shape(name))
            ELAI_lat_median = np.nanpercentile(name, 50, axis=2)
            ELAI_lon_median1 = np.nanpercentile(ELAI_lat_median, 50, axis=2)
            ELAI_lon_median = np.nanpercentile(ELAI_lon_median1, 50, axis=0)

        else:
            ELAI_lat_median = np.nanpercentile(name, 50, axis=1)
            ELAI_lon_median = np.nanpercentile(ELAI_lat_median, 50, axis=1)

        yy = ELAI_lon_median
        xx = np.arange(year[v][0],year[v][1],1)
        x = xx[~np.isnan(yy)]
        y = yy[~np.isnan(yy)]
        print(x,y)
        ax.plot(x, y, '-', color=color[v], label=label[v])

    ax.set_xticks([1982,2000,2015])
    ax.set_xticklabels([1983,2001,2016])

    ax1 = ax.twinx()
    for v, name in zip(range(2), [NDVI3g, KNDVImodis]):
        ELAI_lat_median = np.nanpercentile(name, 50, axis=1)
        ELAI_lon_median = np.nanpercentile(ELAI_lat_median, 50, axis=1)

        yy = ELAI_lon_median
        xx = np.arange(year[v+2][0],year[v+2][1],1)
        x = xx[~np.isnan(yy)]
        y = yy[~np.isnan(yy)]
        print(x,y)
        ax1.plot(x, y, '--', color=color[v+2], label=label[v+2])

    ax1.set_xticks([1982,2000,2015])
    ax1.set_xticklabels([1983,2001,2016])

    return(ax, ax1)

if __name__ == '__main__':
    log_string = 'data-processing :'

    # create figures
    fig = plt.figure(figsize=[4, 2], dpi=300, tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=1.2)

    ####### draw temporal variations of sensitivity for obs, models
    name = ['NDVI3g', 'LAImodis', 'KNDVImodis']
    list = [['study2/results/result_july/NDVI3g-ESM_6v_monthly_1982_2014_0d50_ContributionSensitivity_3Yblock.npy',
             'study2/results/result_july/NDVI3g-ESM_6v_monthly_1982_2014_0d50_ContributionSensitivity_allYear.npy'],
            ['study2/results/result_july/LAImodis-ESM_6v_monthly_2000_2017_0d50_ContributionSensitivity_3Yblock.npy',
             'study2/results/result_july/LAImodis-ESM_6v_monthly_2000_2017_0d50_ContributionSensitivity_allYear.npy'],
            ['study2/results/result_july/KNDVImodis-ESM_6v_monthly_2000_2017_0d50_ContributionSensitivity_3Yblock.npy',
             'study2/results/result_july/KNDVImodis-ESM_6v_monthly_2000_2017_0d50_ContributionSensitivity_allYear.npy']]
    for v in range(3):
        Sensi_3Y_obs = read_data(data_path(list[v][0]))
        Sensi_all_obs = read_data(data_path(list[v][1]))
        Temporal_vari_sensi_n = temporal_vari_sen(Sensi_3Y_obs, Sensi_all_obs)
        np.save(data_path('study2/results/result_july/Temporal_vari_sensi_' + name[v] + '_fig3'), Temporal_vari_sensi_n)

    Temporal_vari_sensi_ELAI = read_data(data_path('study2/results/result_july/Temporal_vari_sensi_ELAI_fig3.npy'))
    Temporal_vari_sensi_NDVI3g = read_data(data_path('study2/results/result_july/Temporal_vari_sensi_NDVI3g_fig3.npy'))
    Temporal_vari_sensi_LAImodis = read_data(data_path('study2/results/result_july/Temporal_vari_sensi_LAImodis_fig3.npy'))
    Temporal_vari_sensi_KNDVImodis = read_data(data_path('study2/results/result_july/Temporal_vari_sensi_KNDVImodis_fig3.npy'))

    ax0 = fig.add_subplot(gs[0])
    ax0, ax01 = plot_with_interquartile(ax0, Temporal_vari_sensi_ELAI[:,0,:,:,:], Temporal_vari_sensi_NDVI3g[0,:,:,:], Temporal_vari_sensi_LAImodis[0,:,:,:], Temporal_vari_sensi_KNDVImodis[0,:,:,:])
    ax0.set_ylabel(tit1[0] + unit2)
    ax01.set_ylabel(tit2[0] + unit2)
    ax0.set_xlabel('Year')
    # ax0.set_ylim(0,0.006)
    # ax01.set_ylim(-0.0001,0.0015)
    # ax0.set_yticks([0, 0.002, 0.004, 0.006])
    # ax01.set_yticks([0, 0.0005, 0.001, 0.0015])
    ax0.legend(loc='upper left', frameon=False)
    ax01.legend(loc='lower left', frameon=False)

    ax1 = fig.add_subplot(gs[1])
    ax1, ax11 = plot_with_interquartile(ax1, Temporal_vari_sensi_ELAI[:,1,:,:,:], Temporal_vari_sensi_NDVI3g[1,:,:,:], Temporal_vari_sensi_LAImodis[1,:,:,:], Temporal_vari_sensi_KNDVImodis[1,:,:,:])
    ax1.set_ylabel(tit1[1] + unit2)
    ax11.set_ylabel(tit2[1] + unit2)
    ax1.set_xlabel('Year')
    # ax1.set_ylim(0,0.001)
    # ax11.set_ylim(0,0.0002)
    # ax1.set_yticks([0, 0.0005, 0.001])
    # ax11.set_yticks([0, 0.0001, 0.0002])

    plt.savefig(data_path('study2/results/result_april/figure2/figs8.jpg'), bbox_inches='tight')

    print('end')

    ########this output is wrong. all-year 3-year sensitivity should be rn again.