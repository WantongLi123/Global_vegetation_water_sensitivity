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
    print(area)

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
    ax.set_yticklabels([0,0.3], fontsize=6)
    ax.set_ylabel('Area ratios', fontsize=6)
    return()

def temporal_vari_sen(n, Sensi_3Y, Sensi_all):
    year = np.shape(Sensi_3Y[0, 2, :, :, :, :])[0]
    temporal_vari_sensi = np.zeros((n, 2, year, 360, 720)) * np.nan
    for v, index in zip([1, 2], [0, 1]):
        for model in range(n):
            sensiSlope_3Y = Sensi_3Y[model, 2, :, v, :, :]
            sensiSlope_all = Sensi_all[model, 3, v, :, :]
            sensiPvalue_all = Sensi_all[model, 4, v, :, :]
            sensiSlope_all = np.repeat(sensiSlope_all[np.newaxis, :, :], year, axis=0)
            sensiPvalue_all = np.repeat(sensiPvalue_all[np.newaxis, :, :], year, axis=0)
            sensiSlope_3Y[sensiSlope_all <= 0] = np.nan
            sensiSlope_3Y[sensiPvalue_all >= 0.05] = np.nan
            temporal_vari_sensi[model, index, :, :, :] = sensiSlope_3Y
            temporal_vari_sensi[model, index, :, :, :] = sensiSlope_3Y
    return(temporal_vari_sensi)

def plot_with_interquartile(ax, ELAI, TrendyS3):
    #draw obs line with interquantile
    ELAI_lat_median = np.nanpercentile(ELAI, 50, axis=2)
    ELAI_lon_median = np.nanpercentile(ELAI_lat_median, 50, axis=2)
    ELAI_model_mean = np.nanpercentile(ELAI_lon_median, 50, axis=0)
    ELAI_model_25th = np.nanpercentile(ELAI_lon_median, 25, axis=0)
    ELAI_model_75th = np.nanpercentile(ELAI_lon_median, 75, axis=0)

    yy = ELAI_model_mean
    yy_25th = ELAI_model_25th
    yy_75th = ELAI_model_75th

    xx = np.arange(1982,2018,1)
    x = xx[~np.isnan(yy)]
    y = yy[~np.isnan(yy)]
    y_25th = yy_25th[~np.isnan(yy)]
    y_75th = yy_75th[~np.isnan(yy)]

    ax.plot(x, y, '-', color='blue', label='obs')
    ax.fill_between(x, y_25th, y_75th, facecolor='blue', alpha=0.2)
    slope1,p1 = mk.original_test(y).slope, mk.original_test(y).p

    #draw model line with interquantile
    TrendyS3_lat_median = np.nanpercentile(TrendyS3, 50, axis=2)
    TrendyS3_lon_median = np.nanpercentile(TrendyS3_lat_median, 50, axis=2)
    TrendyS3_model_mean = np.nanpercentile(TrendyS3_lon_median, 50, axis=0)
    TrendyS3_model_25th = np.nanpercentile(TrendyS3_lon_median, 25, axis=0)
    TrendyS3_model_75th = np.nanpercentile(TrendyS3_lon_median, 75, axis=0)

    yy = TrendyS3_model_mean
    yy_25th = TrendyS3_model_25th
    yy_75th = TrendyS3_model_75th

    xx = np.arange(1982, 2018, 1)
    x = xx[~np.isnan(yy)]
    y = yy[~np.isnan(yy)]
    y_25th = yy_25th[~np.isnan(yy)]
    y_75th = yy_75th[~np.isnan(yy)]

    ax.plot(x, y, '-', color='darkorange', label='Model')
    ax.fill_between(x, y_25th, y_75th, facecolor='darkorange', alpha=0.2)
    slope2,p2 = mk.original_test(y).slope, mk.original_test(y).p

    ax.set_xticks([1982,2000,2015])
    ax.set_xticklabels([1983,2001,2016])

    return(slope1, slope2, p1, p2)

if __name__ == '__main__':
    log_string = 'data-processing :'

    Sensi_3Y_trend_slope = np.zeros((4,360,720)) * np.nan
    Sensi_3Y_trend_p = np.zeros((4,360,720)) * np.nan
    Sensi_3Y_trend_slope[0:2,:,:] = read_data(data_path('study2/results/result_july/ELAI_SM12_ERA5-land_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy'))[0,:,:,:]
    Sensi_3Y_trend_p[0:2,:,:] = read_data(data_path('study2/results/result_july/ELAI_SM12_ERA5-land_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy'))[1,:,:,:]
    Sensi_3Y_trend_slope[2:4, :, :] = read_data(data_path('study2/results/result_july/TRENDY_S3_SM12_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy'))[0,:,:,:]
    Sensi_3Y_trend_p[2:4, :, :] = read_data(data_path('study2/results/result_july/TRENDY_S3_SM12_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy'))[1,:,:,:]
    Sensi_3Y_trend_p[0:2, :, :][np.isnan(Sensi_3Y_trend_p[2:4, :, :])] = np.nan
    Sensi_3Y_trend_p[2:4, :, :][np.isnan(Sensi_3Y_trend_p[0:2, :, :])] = np.nan
    LAI = np.nanmean(read_data(data_path('study2/original_data/LAI/ELAI_4mean_1982_2018_monthly_0.5.npy')),axis=0)

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
    fig = plt.figure(figsize=[6, 6], dpi=300, tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.03)
    gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.02)
    gs01 = gridspec.GridSpecFromSubplotSpec(7, 1, height_ratios=[1,1,0.25,1,1,0.05,0.12], subplot_spec=gs[1], hspace=0)

    label = ['Irrigated/\nnon-vegetated', 'Non-SM\ncontrolled', 'Decreasing\ntrends', 'Increasing\ntrends']
    color_list = plt.get_cmap('coolwarm', 6)
    colorlist = ['#cccccc', '#999999']
    for color in [2, 0, 3, 5]:
        colorlist.append(color_list(color))
    cmap = matplotlib.colors.ListedColormap(colorlist)

    ########## draw global maps of sensitivity trends for obs, models
    ax00 = fig.add_subplot(gs01[0])
    graph(ax00, sen_trend[0, 30:300, :], cmap)
    ax00.set_title('Trends of ' + tit[0] + unit1)
    ax01 = plt.axes([0.46, 0.72, 0.06, 0.06])  # [left,bottom,width,height]
    ax01.get_xaxis().set_ticks([])
    ax01.spines['right'].set_visible(False)
    ax01.spines['top'].set_visible(False)
    bar_plot(sen_trend[0,:,:], ax01)
    #
    ax10 = fig.add_subplot(gs01[1])
    graph(ax10, sen_trend[2, 30:300, :], cmap)
    ax11 = plt.axes([0.46, 0.545, 0.06, 0.06])  # [left,bottom,width,height]
    ax11.get_xaxis().set_ticks([])
    ax11.spines['right'].set_visible(False)
    ax11.spines['top'].set_visible(False)
    bar_plot(sen_trend[2, :, :], ax11)

    ax20 = fig.add_subplot(gs01[3])
    graph(ax20, sen_trend[1, 30:300, :], cmap)
    ax20.set_title('Trends of ' + tit[1] + unit1)
    ax21 = plt.axes([0.46, 0.325, 0.06, 0.06])  # [left,bottom,width,height]
    ax21.get_xaxis().set_ticks([])
    ax21.spines['right'].set_visible(False)
    ax21.spines['top'].set_visible(False)
    bar_plot(sen_trend[1, :, :], ax21)

    ax30 = fig.add_subplot(gs01[4])
    graph(ax30, sen_trend[3, 30:300, :], cmap)
    ax31 = plt.axes([0.46, 0.155, 0.06, 0.06])  # [left,bottom,width,height]
    ax31.get_xaxis().set_ticks([])
    ax31.spines['right'].set_visible(False)
    ax31.spines['top'].set_visible(False)
    bar_plot(sen_trend[3, :, :], ax31)

    ax40 = fig.add_subplot(gs01[6])
    norm = colors.Normalize(vmin=0, vmax=7)
    color_list = plt.get_cmap('coolwarm', 6)
    colorlist = ['#cccccc','#cccccc','#999999','#999999']
    for color in [2, 0, 3, 5]:
        colorlist.append(color_list(color))
    cmap = matplotlib.colors.ListedColormap(colorlist)
    matplotlib.colorbar.ColorbarBase(ax40, ticks=[0.125,0.375,0.625,0.875], cmap=cmap, orientation='horizontal')
    ax40.set_xticklabels(label)



    # ######### draw temporal variations of sensitivity for obs, models
    # Sensi_3Y_obs = read_data(data_path('study2/results/result_july/Ensemble-8obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_3Yblock_singleSMOUT.npy'))
    # Sensi_all_obs = read_data(data_path('study2/results/result_july/Ensemble-8obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_singleSMOUT.npy'))
    # Sensi_3Y_TrendyS3 = read_data(data_path('study2/results/result_april/TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_3Yblock_SM13modelOUT.npy'))
    # Sensi_all_TrendyS3 = read_data(data_path('study2/results/result_april/TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_SM13modelOUT.npy'))
    # Temporal_vari_sensi_ELAI = temporal_vari_sen(8, Sensi_3Y_obs, Sensi_all_obs)
    # Temporal_vari_sensi_TrendyS3 = temporal_vari_sen(10, Sensi_3Y_TrendyS3, Sensi_all_TrendyS3)
    # np.save(data_path('study2/results/result_july/Temporal_vari_sensi_ELAI_fig3'), Temporal_vari_sensi_ELAI)
    # np.save(data_path('study2/results/result_july/Temporal_vari_sensi_TrendyS3_fig3'), Temporal_vari_sensi_TrendyS3)

    Temporal_vari_sensi_ELAI = read_data(data_path('study2/results/result_july/Temporal_vari_sensi_ELAI_fig3.npy'))
    Temporal_vari_sensi_TrendyS3 = read_data(data_path('study2/results/result_july/Temporal_vari_sensi_TrendyS3_fig3.npy'))
    ax0 = fig.add_subplot(gs00[0])
    slope1, slope2, p1, p2 = plot_with_interquartile(ax0, Temporal_vari_sensi_ELAI[:,0,:,:,:], Temporal_vari_sensi_TrendyS3[:,0,:,:,:])
    ax0.set_ylabel(tit[0] + unit2)
    ax0.get_xaxis().set_ticks([])
    ax0.set_ylim(0,0.006)
    ax0.set_yticks([0, 0.0025, 0.005])
    ax0.legend(loc='lower right', frameon=False)
    print('slope:', np.round(slope1,5),np.round(slope2,5))
    print('p:', np.round(p1,5),np.round(p2,5))
    ax0.text(1982, 0.0055, '1.5e-04**', color='blue')
    ax0.text(1982, 0.005, '-7e-05', color='darkorange')

    ax1 = fig.add_subplot(gs00[1])
    slope1, slope2, p1, p2 = plot_with_interquartile(ax1, Temporal_vari_sensi_ELAI[:,1,:,:,:], Temporal_vari_sensi_TrendyS3[:,1,:,:,:])
    ax1.set_ylabel(tit[1] + unit2)
    ax1.set_xlabel('Year')
    ax1.set_ylim(0,0.003)
    ax1.set_yticks([0, 0.001, 0.002])
    print('slope:', np.round(slope1,5),np.round(slope2,5))
    print('p:', np.round(p1,5),np.round(p2,5))
    ax1.text(1982, 0.0028, '1e-05**', color='blue')
    ax1.text(1982, 0.0025, '-6e-05**', color='darkorange')
    ax1.text(1982, 0.0022, '($\mathregular{mm^{-1}}$ per 3 years)')
    # p<=0.01, p>0.1, p<=0.01, p>0.1

    ax1.text(1965,0.0059, '(A)')
    ax1.text(2019, 0.0059, '(C) obs')
    ax1.text(2019, 0.0045, '(D) Model')
    ax1.text(1965, 0.0028, '(B)')
    ax1.text(2019, 0.0028, '(E) obs')
    ax1.text(2019, 0.0014, '(F) Model')

    plt.savefig(data_path('study2/results/result_april/figure2/fig4.jpg'), bbox_inches='tight')

    print('end')