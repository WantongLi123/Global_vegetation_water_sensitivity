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
# matplotlib.font_manager._rebuild()

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
plt.rcParams.update({'font.size': 5})
plt.rcParams['axes.unicode_minus']=False

tit = [r'$\frac{\partial{LAI}}{\partial{SMnear}}$',r'$\frac{\partial{LAI}}{\partial{SMsub}}$']
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

def trend_cal(Sensi_3Y, Sensi_all, Sensi_all_ELAI,Sensi_all_TRENDY):
    trend = np.zeros((2,2,360,720),dtype=np.float32)
    trend[:] = np.nan
    for v, index in zip([1, 2], [0,1]):
        sensiSlope_3Y = Sensi_3Y[2, :, v, :, :]
        sensiSlope_all = Sensi_all[3, v, :, :]
        sensiPvalue_all = Sensi_all[4, v, :, :]
        sensiSlope_all = np.repeat(sensiSlope_all[np.newaxis, :, :], np.shape(sensiSlope_3Y)[0], axis=0)
        sensiPvalue_all = np.repeat(sensiPvalue_all[np.newaxis, :, :], np.shape(sensiSlope_3Y)[0], axis=0)
        sensiSlope_3Y[sensiSlope_all < 0] = np.nan
        sensiSlope_3Y[sensiPvalue_all > 0.01] = np.nan

        sensiSlope_all_ELAI = Sensi_all_ELAI[3, v, :, :]
        sensiPvalue_all_ELAI = Sensi_all_ELAI[4, v, :, :]
        sensiSlope_all_ELAI = np.repeat(sensiSlope_all_ELAI[np.newaxis, :, :], np.shape(sensiSlope_3Y)[0], axis=0)
        sensiPvalue_all_ELAI = np.repeat(sensiPvalue_all_ELAI[np.newaxis, :, :], np.shape(sensiSlope_3Y)[0], axis=0)
        sensiSlope_3Y[sensiSlope_all_ELAI < 0] = np.nan
        sensiSlope_3Y[sensiPvalue_all_ELAI > 0.01] = np.nan

        sensiSlope_all_TRENDY = Sensi_all_TRENDY[3, v, :, :]
        sensiPvalue_all_TRENDY = Sensi_all_TRENDY[4, v, :, :]
        sensiSlope_all_TRENDY = np.repeat(sensiSlope_all_TRENDY[np.newaxis, :, :], np.shape(sensiSlope_3Y)[0], axis=0)
        sensiPvalue_all_TRENDY = np.repeat(sensiPvalue_all_TRENDY[np.newaxis, :, :], np.shape(sensiSlope_3Y)[0], axis=0)
        sensiSlope_3Y[sensiSlope_all_TRENDY < 0] = np.nan
        sensiSlope_3Y[sensiPvalue_all_TRENDY > 0.01] = np.nan

        for row in range(360):
            for col in range(720):
                sensi = sensiSlope_3Y[:, row, col]
                y = sensi[~np.isnan(sensi)]
                if len(y) >=5:
                    result = mk.original_test(y)
                    trend[0, index, row, col] = result.slope
                    trend[1, index, row, col] = result.p
                    # print(y,trend[0, index, row, col],trend[1, index, row, col])
    return(trend)

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
    # print(area_2/study_area,area_3/study_area,area_4/study_area,area_5/study_area)

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
    ax.set_ylabel('Area fraction', fontsize=6)
    return()

def temporal_vari_sen(n, Sensi_3Y, Sensi_all, Sensi_all_ELAI, Sensi_all_TRENDY):
    year = np.shape(Sensi_3Y[0, 2, :, :, :, :])[0]
    temporal_vari_sensi = np.zeros((n, 2, year, 360, 720)) * np.nan
    for v, index in zip([1, 2], [0, 1]):
        for model in range(n):
            sensiSlope_3Y = Sensi_3Y[model, 2, :, v, :, :]
            sensiSlope_all = Sensi_all[model, 3, v, :, :]
            sensiPvalue_all = Sensi_all[model, 4, v, :, :]
            sensiSlope_all = np.repeat(sensiSlope_all[np.newaxis, :, :], year, axis=0)
            sensiPvalue_all = np.repeat(sensiPvalue_all[np.newaxis, :, :], year, axis=0)
            sensiSlope_3Y[sensiSlope_all < 0] = np.nan
            sensiSlope_3Y[sensiPvalue_all > 0.01] = np.nan

            sensiSlope_all_ELAI = Sensi_all_ELAI[3, v, :, :]
            sensiPvalue_all_ELAI = Sensi_all_ELAI[4, v, :, :]
            sensiSlope_all_ELAI = np.repeat(sensiSlope_all_ELAI[np.newaxis, :, :], year, axis=0)
            sensiPvalue_all_ELAI = np.repeat(sensiPvalue_all_ELAI[np.newaxis, :, :], year, axis=0)
            sensiSlope_3Y[sensiSlope_all_ELAI < 0] = np.nan
            sensiSlope_3Y[sensiPvalue_all_ELAI > 0.01] = np.nan

            sensiSlope_all_TRENDY = Sensi_all_TRENDY[3, v, :, :]
            sensiPvalue_all_TRENDY = Sensi_all_TRENDY[4, v, :, :]
            sensiSlope_all_TRENDY = np.repeat(sensiSlope_all_TRENDY[np.newaxis, :, :], year,axis=0)
            sensiPvalue_all_TRENDY = np.repeat(sensiPvalue_all_TRENDY[np.newaxis, :, :], year,axis=0)
            sensiSlope_3Y[sensiSlope_all_TRENDY < 0] = np.nan
            sensiSlope_3Y[sensiPvalue_all_TRENDY > 0.01] = np.nan

            temporal_vari_sensi[model, index, :, :, :] = sensiSlope_3Y

    return(temporal_vari_sensi)

def plot_with_interquartile(ax, ELAI, TrendyS3):
    #draw obs line with interquantile
    ELAI_lat_median = np.nanpercentile(ELAI, 50, axis=2)
    ELAI_lon_median = np.nanpercentile(ELAI_lat_median, 50, axis=2)
    ELAI_lon_median_from1982 = np.zeros((4,36))
    for v in range(4):
        ELAI_lon_median_from1982[v,:] = ELAI_lon_median[v,:] - ELAI_lon_median[v,0]
        # yy = ELAI_lon_median_from1982[v,:]
        # xx = np.arange(1982, 2018, 1)
        # x = xx[~np.isnan(yy)]
        # y = yy[~np.isnan(yy)]
        # ax.plot(x, y, '--', color='black', label='', linewidth=0.1)

    ELAI_model_mean = np.nanpercentile(ELAI_lon_median_from1982, 50, axis=0)
    ELAI_model_25th = np.nanpercentile(ELAI_lon_median_from1982, 25, axis=0)
    ELAI_model_75th = np.nanpercentile(ELAI_lon_median_from1982, 75, axis=0)

    yy = ELAI_model_mean
    yy_25th = ELAI_model_25th
    yy_75th = ELAI_model_75th

    xx = np.arange(1982,2018,1)
    x = xx[~np.isnan(yy)]
    y = yy[~np.isnan(yy)]
    y_25th = yy_25th[~np.isnan(yy)]
    y_75th = yy_75th[~np.isnan(yy)]

    ax.plot(x, y, '-', color='black', label='Obs')
    ax.fill_between(x, y_25th, y_75th, facecolor='black', alpha=0.2)
    slope1,p1 = mk.original_test(y).slope, mk.original_test(y).p

    #draw model line with interquantile
    TrendyS3_lat_median = np.nanpercentile(TrendyS3, 50, axis=2)
    TrendyS3_lon_median = np.nanpercentile(TrendyS3_lat_median, 50, axis=2)
    TrendyS3_lon_median_from1982 = np.zeros((8,36))
    for v in range(8):
        TrendyS3_lon_median_from1982[v,:] = TrendyS3_lon_median[v,:] - TrendyS3_lon_median[v,0]
        # yy = TrendyS3_lon_median_from1982[v, :]
        # xx = np.arange(1982, 2018, 1)
        # x = xx[~np.isnan(yy)]
        # y = yy[~np.isnan(yy)]
        # ax.plot(x, y, '--', color='darkorange', label='', linewidth=0.1)

    TrendyS3_model_mean = np.nanpercentile(TrendyS3_lon_median_from1982, 50, axis=0)
    TrendyS3_model_25th = np.nanpercentile(TrendyS3_lon_median_from1982, 25, axis=0)
    TrendyS3_model_75th = np.nanpercentile(TrendyS3_lon_median_from1982, 75, axis=0)

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

    Sensi_3Y = read_data(data_path('study2/results/result_oct/Ensemble-6obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_3Yblock.npy'))
    Sensi_all = read_data(data_path('study2/results/result_oct/Ensemble-6obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear.npy'))
    Sensi_all_ELAI = read_data(data_path('study2/results/result_oct/Ensemble-6obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear.npy'))[5,:,:, :, :]
    Sensi_all_TRENDY = read_data(data_path('study2/results/result_april/TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear.npy'))[9, :, :, :, :]

    # Trend = trend_cal(Sensi_3Y, Sensi_all, Sensi_all_ELAI,Sensi_all_TRENDY)
    # np.save(data_path('study2/results/result_oct/VOD_SM12_ERA5-land_monthly_1988_2017_0d50_3Yblock_SensiTrend'), Trend)
    # Trend = np.zeros((6,2,2,360,720),dtype=np.float32)
    # for v in range(6):
    #     Trend[v,:,:,:,:] = trend_cal(Sensi_3Y[v,:,:,:, :, :], Sensi_all[v,:,:, :, :], Sensi_all_ELAI, Sensi_all_TRENDY)
    # np.save(data_path('study2/results/result_oct/ELAI_SM12_ERA5-land_monthly_1982_2017_0d50_3Yblock_SensiTrend'), Trend)

    ind_LAI = 5
    Sensi_3Y_trend_slope = np.zeros((2,360,720)) * np.nan
    Sensi_3Y_trend_p = np.zeros((2,360,720)) * np.nan
    Sensi_3Y_trend_slope[0,:,:] = read_data(data_path('study2/results/result_oct/ELAI_SM12_ERA5-land_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy'))[ind_LAI, 0,1,:,:] #[0,0,:,:] for surf
    Sensi_3Y_trend_p[0,:,:] = read_data(data_path('study2/results/result_oct/ELAI_SM12_ERA5-land_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy'))[ind_LAI, 1,1,:,:] #[1,0,:,:] for surf
    Sensi_3Y_trend_slope[1, :, :] = read_data(data_path('study2/results/result_july/TRENDY_S3_SM12_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy'))[0,1,:,:] #[0,0,:,:] for surf
    Sensi_3Y_trend_p[1, :, :] = read_data(data_path('study2/results/result_july/TRENDY_S3_SM12_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy'))[1,1,:,:] #[1,0,:,:] for surf
    Sensi_3Y_trend_p[0, :, :][np.isnan(Sensi_3Y_trend_p[1, :, :])] = np.nan
    Sensi_3Y_trend_p[1, :, :][np.isnan(Sensi_3Y_trend_p[0, :, :])] = np.nan
    LAI = np.nanmean(read_data(data_path('study2/original_data/LAI/EnLAI_5mean_1982_2018_monthly_0.5.npy')),axis=0)

    # mask VCF<5%, irrigation>10%
    irrigation = read_data(data_path('study2/original_data/irrigation/gmia_v5_aei_pct_360_720.npy'))
    fvc = read_data(data_path('Proj1VD/original_data/Landcover/VCF5KYR/vcf5kyr_v001/VCF_1982_to_2016_0Tree_1nonTree_yearly.npy'))
    vegetation_cover = np.nanmean(fvc[0, :, :, :] + fvc[1, :, :, :], axis=0)

    sen_trend = np.zeros((2, 360, 720)) * np.nan
    for v in range(2):
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

    # create figures
    fig = plt.figure(figsize=[4, 2.2], dpi=300, tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.03) # distinguish two columns for A and B,C
    gs00 = gridspec.GridSpecFromSubplotSpec(4, 1, height_ratios=[1,1,0.05,0.12], subplot_spec=gs[1], hspace=0) # distinguish 4 rows for B,C

    label = ['Irrigated/\nnon-vegetated', 'Non-soil-moisture\ncontrolled', 'Decreasing\ntrends', 'Increasing\ntrends']
    color_list = plt.get_cmap('coolwarm', 6)
    colorlist = ['#cccccc', '#999999']
    for color in [2, 0, 3, 5]:
        colorlist.append(color_list(color))
    cmap = matplotlib.colors.ListedColormap(colorlist)

    ########## draw global maps of sensitivity trends for obs, models
    ax00 = fig.add_subplot(gs00[0])
    graph(ax00, sen_trend[0, 30:300, :], cmap)
    ax00.set_title('Trends of ' + tit[1] + unit1) # change here if near-surf
    ax01 = plt.axes([0.482, 0.6, 0.03, 0.08])  # [left,bottom,width,height]
    ax01.get_xaxis().set_ticks([])
    ax01.spines['right'].set_visible(False)
    ax01.spines['top'].set_visible(False)
    bar_plot(sen_trend[0,:,:], ax01)

    ax10 = fig.add_subplot(gs00[1])
    graph(ax10, sen_trend[1, 30:300, :], cmap)
    ax11 = plt.axes([0.482, 0.25, 0.03, 0.08])  # [left,bottom,width,height]
    ax11.get_xaxis().set_ticks([])
    ax11.spines['right'].set_visible(False)
    ax11.spines['top'].set_visible(False)
    bar_plot(sen_trend[1, :, :], ax11)

    ax20 = fig.add_subplot(gs00[3])
    norm = colors.Normalize(vmin=0, vmax=7)
    color_list = plt.get_cmap('coolwarm', 6)
    colorlist = ['#cccccc','#cccccc','#999999','#999999']
    for color in [2, 0, 3, 5]:
        colorlist.append(color_list(color))
    cmap = matplotlib.colors.ListedColormap(colorlist)
    matplotlib.colorbar.ColorbarBase(ax20, ticks=[0.125,0.375,0.625,0.875], cmap=cmap, orientation='horizontal')
    ax20.set_xticklabels(label)

    ######### draw temporal variations of sensitivity for obs, models
    # Sensi_3Y_obs = read_data(data_path('study2/results/result_oct/Ensemble-6obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_3Yblock.npy'))
    # Sensi_all_obs = read_data(data_path('study2/results/result_oct/Ensemble-6obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear.npy'))
    # Sensi_3Y_TrendyS3 = read_data(data_path('study2/results/result_april/TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_3Yblock_SM13modelOUT.npy'))
    # Sensi_all_TrendyS3 = read_data(data_path('study2/results/result_april/TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_SM13modelOUT.npy'))
    # Temporal_vari_sensi_ELAI = temporal_vari_sen(6, Sensi_3Y_obs, Sensi_all_obs, Sensi_all_ELAI, Sensi_all_TRENDY)
    # # Temporal_vari_sensi_TrendyS3 = temporal_vari_sen(9, Sensi_3Y_TrendyS3, Sensi_all_TrendyS3, Sensi_all_ELAI, Sensi_all_TRENDY)
    # np.save(data_path('study2/results/result_oct/Temporal_vari_sensi_ELAI_fig3'), Temporal_vari_sensi_ELAI)
    # # np.save(data_path('study2/results/result_oct/Temporal_vari_sensi_TrendyS3_fig3'), Temporal_vari_sensi_TrendyS3)

    Temporal_vari_sensi_ELAI = read_data(data_path('study2/results/result_oct/Temporal_vari_sensi_ELAI_fig3.npy'))
    Temporal_vari_sensi_TrendyS3 = read_data(data_path('study2/results/result_july/Temporal_vari_sensi_TrendyS3_fig3.npy'))
    ax1 = fig.add_subplot(gs[0])
    slope1, slope2, p1, p2 = plot_with_interquartile(ax1, Temporal_vari_sensi_ELAI[:,1,:,:,:], Temporal_vari_sensi_TrendyS3[:,1,:,:,:]) #[:,0,:,:,:] for surf
    ax1.set_ylabel(tit[1] + unit2) # change here if near-surf
    ax1.set_xlabel('Year')
    ax1.set_ylim(-0.0003,0,0.0003)
    ax1.set_yticks([-0.0003,0,0.0003])
    ax1.legend(loc='lower right', frameon=False)
    print('slope:', slope1,slope2)
    print('p:', np.round(p1,5),np.round(p2,5))
    eee='$\mathregular{x10^{-6}}$'
    ax1.text(1982, 0.00026, '8.0'+eee+'**', color='black')
    ax1.text(1982, 0.00022, '-4.0'+eee, color='darkorange')
    ax1.text(1982, 0.00018, '($\mathregular{mm^{-1}}$ per 3 years)')
    # p<0.01, p>0.1, p<0.01, p<0.1

    ax1.text(1962, 0.00028, '(a)')
    ax1.text(2019, 0.00028, '(b) Obs')
    ax1.text(2019, 0, '(c) Model')

    plt.savefig(data_path('study2/results/result_oct/figure2/fig3.jpg'), bbox_inches='tight')

    print('end')