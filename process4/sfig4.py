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
plt.rcParams.update({'font.size': 8})
plt.rcParams['axes.unicode_minus']=False

tit = [r'$\frac{\partial{LAI}}{\partial{SMsurf}}$',r'$\frac{\partial{LAI}}{\partial{SMroot}}$']
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

def graph(ax, target, cmap):
    cs = ax.pcolor(target[::-1,:], cmap=cmap)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    return(cs)

def bar_plot(impor,ax):
    # convert each degree of lat and lon to KM2
    lat_area = np.ones((360,720))/2*110.574
    lon_area = np.ones((360,720))
    for row in range(360):
        if row <= 180:
            lat_ind = 90 - row / 2
            lon_area[row, :] = 111.320 * math.cos(math.radians(lat_ind))/2
            # print('lat_ind', lat_ind, 'lon_area', lon_area[row, 0])
        if row > 180:
            lat_ind = row / 2 - 90
            lon_area[row, :] = 111.320 * math.cos(math.radians(lat_ind))/2
            # print('lat_ind', lat_ind, 'lon_area', lon_area[row, 0])
    area = lat_area * lon_area

    color_list = plt.get_cmap('coolwarm', 6)
    ind = np.arange(1)  # the x locations for the groups
    width = 0.3  # the width of the bars: can also be len(x) sequence
    study_area = np.nansum(area[np.where((impor > 0) & (impor <= 5))])
    area_2 = np.nansum(area[np.where(impor == 2)])
    area_3 = np.nansum(area[np.where(impor == 3)])
    area_4 = np.nansum(area[np.where(impor == 4)])
    area_5 = np.nansum(area[np.where(impor == 5)])
    print(area_2/study_area,area_3/study_area,area_4/study_area,area_5/study_area)

    prop2 = []
    prop2.append(np.round(area_2 / study_area, 3))
    plt.bar(ind-0.15, prop2, width, color=color_list(2))

    prop3 = []
    prop3.append(np.round(area_3 / study_area, 3))
    plt.bar(ind-0.15, prop3, width, color=color_list(0), bottom=prop2)

    prop4 = []
    prop4.append(np.round(area_4 / study_area, 3))
    plt.bar(ind+0.15, prop4, width, color=color_list(3))

    prop5 = []
    prop5.append(np.round(area_5 / study_area, 3))
    plt.bar(ind + 0.15, prop5, width, color=color_list(5), bottom=prop4)
    ax.set_yticks([0,0.3])
    ax.set_yticklabels([0,0.3])
    ax.set_ylabel('Area ratios')
    return()

if __name__ == '__main__':
    log_string = 'data-processing :'

    Sensi_3Y_trend_slope = np.zeros((4, 360, 720)) * np.nan
    Sensi_3Y_trend_p = np.zeros((4, 360, 720)) * np.nan
    Sensi_3Y_trend_slope[0:2, :, :] = read_data(
        data_path('study2/results/result_july/ELAI_SM12_ERA5-land_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy'))[0, :,
                                      :, :]
    Sensi_3Y_trend_p[0:2, :, :] = read_data(
        data_path('study2/results/result_july/ELAI_SM12_ERA5-land_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy'))[1, :,
                                  :, :]
    Sensi_3Y_trend_slope[2:4, :, :] = read_data(
        data_path('study2/results/result_july/TRENDY_S3_SM12_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy'))[0, :, :,
                                      :]
    Sensi_3Y_trend_p[2:4, :, :] = read_data(
        data_path('study2/results/result_july/TRENDY_S3_SM12_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy'))[1, :, :,
                                  :]
    LAI = np.nanmean(read_data(data_path('study2/original_data/LAI/ELAI_4mean_1982_2018_monthly_0.5.npy')), axis=0)

    # mask VCF<5%, irrigation>10%
    irrigation = read_data(data_path('study2/original_data/irrigation/gmia_v5_aei_pct_360_720.npy'))
    fvc = read_data(data_path('Proj1VD/original_data/Landcover/VCF5KYR/vcf5kyr_v001/VCF_1982_to_2016_0Tree_1nonTree_yearly.npy'))
    vegetation_cover = np.nanmean(fvc[0, :, :, :] + fvc[1, :, :, :], axis=0)

    sen_trend = np.zeros((4, 360, 720)) * np.nan
    for v in range(4):
        sen_trend[v, :, :][LAI > 0] = 0
        for row in range(360):
            for col in range(720):
                if LAI[row,col]>0 and np.isnan(Sensi_3Y_trend_p[v, row, col]) and irrigation[row, col] <= 10 and vegetation_cover[row, col] >= 5:
                    sen_trend[v, row, col] = 1
                else:
                    if Sensi_3Y_trend_p[v, row, col] > 0.1 and Sensi_3Y_trend_slope[v, row, col] < 0:
                        sen_trend[v, row, col] = 2
                    elif Sensi_3Y_trend_p[v, row, col] <= 0.1 and Sensi_3Y_trend_slope[v, row, col] < 0:
                        sen_trend[v, row, col] = 3
                    elif Sensi_3Y_trend_p[v, row, col] > 0.1 and Sensi_3Y_trend_slope[v, row, col] > 0:
                        sen_trend[v, row, col] = 4
                    elif Sensi_3Y_trend_p[v, row, col] <= 0.1 and Sensi_3Y_trend_slope[v, row, col] > 0:
                        sen_trend[v, row, col] = 5

        print(np.nanmin(sen_trend[v, :, :]), np.nanmax(sen_trend[v, :, :]))

    # create figures
    fig = plt.figure(figsize=[4, 6], dpi=300, tight_layout=True)
    gs = gridspec.GridSpec(7, 1, height_ratios=[1,1,0.25,1,1,0.05,0.12], hspace=0)

    label = ['Irrigated/\nnon-vegetated', 'Non-SM\ncontrolled', 'Decreasing\ntrends', 'Increasing\ntrends']
    color_list = plt.get_cmap('coolwarm', 6)
    colorlist = ['#cccccc', '#999999']
    for color in [2, 0, 3, 5]:
        colorlist.append(color_list(color))
    cmap = matplotlib.colors.ListedColormap(colorlist)

    ########## draw global maps of sensitivity trends for obs, models
    ax00 = fig.add_subplot(gs[0])
    graph(ax00, sen_trend[0, 30:300, :], cmap)
    ax00.set_title('Trends of ' + tit[0] + unit1)
    ax01 = plt.axes([0.23, 0.724, 0.08, 0.06])  # [left,bottom,width,height]
    ax01.get_xaxis().set_ticks([])
    ax01.spines['right'].set_visible(False)
    ax01.spines['top'].set_visible(False)
    bar_plot(sen_trend[0,:,:], ax01)

    ax10 = fig.add_subplot(gs[1])
    graph(ax10, sen_trend[2, 30:300, :], cmap)
    ax11 = plt.axes([0.23, 0.547, 0.08, 0.06])  # [left,bottom,width,height]
    ax11.get_xaxis().set_ticks([])
    ax11.spines['right'].set_visible(False)
    ax11.spines['top'].set_visible(False)
    bar_plot(sen_trend[2, :, :], ax11)

    ax20 = fig.add_subplot(gs[3])
    graph(ax20, sen_trend[1, 30:300, :], cmap)
    ax20.set_title('Trends of ' + tit[1] + unit1)
    ax21 = plt.axes([0.23, 0.33, 0.08, 0.06])  # [left,bottom,width,height]
    ax21.get_xaxis().set_ticks([])
    ax21.spines['right'].set_visible(False)
    ax21.spines['top'].set_visible(False)
    bar_plot(sen_trend[1, :, :], ax21)

    ax30 = fig.add_subplot(gs[4])
    graph(ax30, sen_trend[3, 30:300, :], cmap)
    ax31 = plt.axes([0.23, 0.156, 0.08, 0.06])  # [left,bottom,width,height]
    ax31.get_xaxis().set_ticks([])
    ax31.spines['right'].set_visible(False)
    ax31.spines['top'].set_visible(False)
    bar_plot(sen_trend[3, :, :], ax31)

    ax40 = fig.add_subplot(gs[6])
    norm = colors.Normalize(vmin=0, vmax=7)
    color_list = plt.get_cmap('coolwarm', 6)
    colorlist = ['#cccccc','#cccccc','#999999','#999999']
    for color in [2, 0, 3, 5]:
        colorlist.append(color_list(color))
    cmap = matplotlib.colors.ListedColormap(colorlist)
    matplotlib.colorbar.ColorbarBase(ax40, ticks=[0.125,0.375,0.625,0.875], cmap=cmap, orientation='horizontal')
    ax40.set_xticklabels(label)

    plt.text(0.015, 36.15, '(A) obs')
    plt.text(0.015, 27.75, '(B) model')
    plt.text(0.015, 17.3, '(C) obs')
    plt.text(0.015, 8.95, '(D) model')

    plt.savefig(data_path('study2/results/result_april/figure2/figs4.jpg'), bbox_inches='tight')

    print('end')