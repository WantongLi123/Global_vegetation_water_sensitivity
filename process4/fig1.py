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
from mpl_toolkits.basemap import Basemap
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
plt.rcParams.update({'font.size': 6})
plt.rcParams['axes.unicode_minus']=False

tit = [r'$\frac{\partial{LAI}}{\partial{SMsurf}}$',r'$\frac{\partial{LAI}}{\partial{SMroot}}$']
unit = ' ($\mathregular{mm^{-1}}$)'

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

def graph(ax, target, **kwargs):
    my_map = Basemap(ax=ax, resolution='l', projection='cyl', llcrnrlon=-180, urcrnrlon=180, llcrnrlat=-60, urcrnrlat=75)
    my_map.drawcoastlines(linewidth=0.2)

    x = np.linspace(my_map.llcrnrlon, my_map.urcrnrlon, target.shape[1])
    y = np.linspace(my_map.llcrnrlat, my_map.urcrnrlat, target.shape[0])
    xx, yy = np.meshgrid(x, y)

    cs = my_map.pcolor(xx, yy, target[::-1,:], **kwargs)

    return(cs)

def heat_map_20(arid, tem, coef,x_axis_20,y_axis_20, x_axis_4, y_axis_4):
    data_20x20 = np.zeros((6, 20, 20), dtype=np.float64) # data[0,:,:]: mean coefficient, data[1,:,:]: median coefficient, data[2,:,:]: couning pixels
    for v in range(6):
        c = coef[v,:,:]
        for y in range(20):
            for x in range(20):
                with np.errstate(invalid='ignore'):
                    mask = c[np.where((arid >= x_axis_20[x]) & (arid <=x_axis_20[x + 1]) &
                                      (tem >= y_axis_20[y]) & (tem <= y_axis_20[y + 1]))]
                    data_20x20[v, y, x] = np.nanmedian(mask)

    data_4x4 = np.zeros((6, 4, 4),dtype=np.float64)  # data[0,:,:]: mean coefficient, data[1,:,:]: median coefficient, data[2,:,:]: couning pixels
    num = np.zeros((6, 4, 4),dtype=np.int)
    for v in range(6):
        c = coef[v, :, :]
        for y in range(4):
            for x in range(4):
                with np.errstate(invalid='ignore'):
                    mask = c[np.where((arid >= x_axis_4[x]) & (arid <= x_axis_4[x + 1]) &
                                      (tem >= y_axis_4[y]) & (tem <= y_axis_4[y + 1]))]
                    data_4x4[v, y, x] = np.nanmedian(mask)
                    num[v,y,x] = np.sum(~np.isnan(mask))
    return(data_20x20, data_4x4,num)

def heatmap_20(ax, data, **kwargs):
    im1 = ax.pcolor(data, **kwargs)

    ax.set_xticks([0,10,20])
    ax.set_yticks([0,10,20])
    ax.set_xticklabels([0, 1, 4])
    ax.set_yticklabels([5,15,30])

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    return(im1)

def heat_map_corr(aridity, t2m, data_obs, data_model,xtick,ytick):
    corr = np.zeros((2, 4, 4))*np.nan
    num = np.zeros((2, 4, 4),dtype=np.int)
    for v in range(2):
        for y in range(4):
            for x in range(4):
                mask_obs = data_obs[v,:,:][np.where((aridity >= xtick[x]) & (aridity <= xtick[x + 1]) & (t2m >= ytick[y]) & (t2m <= ytick[y + 1]))]
                mask_model = data_model[v,:,:][np.where((aridity >= xtick[x]) & (aridity <= xtick[x + 1]) & (t2m >= ytick[y]) & (t2m <= ytick[y + 1]))]
                corr_v1 = mask_obs[~np.isnan(mask_obs) & ~np.isnan(mask_model)]
                corr_v2 = mask_model[~np.isnan(mask_obs) & ~np.isnan(mask_model)]
                num[v,y,x] = np.sum(~np.isnan(corr_v1))
                if num[v,y,x] > 10:
                    if np.isnan(corr_v1).any() or np.isinf(corr_v1).any() or np.all(corr_v1 == 0) or np.isnan(
                            corr_v2).any() or np.isinf(corr_v2).any() or np.all(corr_v2 == 0):
                        corr[v, y, x] = np.nan
                    else:
                        corr[v, y, x], p_value = stats.spearmanr(corr_v1, corr_v2)

    return (corr, num)

def heatmap_corr(ax, data, ticks, **kwargs):
    im1 = ax.pcolor(data, **kwargs)
    ax.set_xticks([0,2,4])
    ax.set_yticks([0,2,4])
    ax.set_xticklabels([0, 1, 4])
    ax.set_yticklabels([5, 15, 30])

    ax.set_xlabel('Aridity')
    cbar = ax.figure.colorbar(im1, ax=ax, ticks=ticks, extend='both', shrink=0.6)
    cbar.set_ticklabels(ticks)

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    return(im1)

def heatmap_num(num, ax, data):
    im1 = ax.pcolor(data,cmap=plt.get_cmap('coolwarm'))
    ax.set_xticks([0, 2, 4])
    ax.set_yticks([0, 2, 4])
    ax.set_xticklabels([0, 1, 4])
    ax.set_yticklabels([5, 15, 30])

    ax.set_xlabel('Aridity')
    ax.set_ylabel('Temperature ($^\circ$C)')

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    for y in range(4):
        for x in range(4):
            ax.text(x+0.5, y+0.5, num[y, x], ha="center", va="center", color="white")
    return (im1)

if __name__ == '__main__':
    log_string = 'Log draw_ndvi.py : '

    # vegetation masks
    irrigation = read_data(data_path('study2/original_data/irrigation/gmia_v5_aei_pct_360_720.npy'))
    fvc = read_data(data_path('Proj1VD/original_data/Landcover/VCF5KYR/vcf5kyr_v001/VCF_1982_to_2016_0Tree_1nonTree_yearly.npy'))
    vegetation_cover = np.nanmean(fvc[0, :, :, :] + fvc[1, :, :, :], axis=0)

    overall_sen = np.zeros((6,360,720)) * np.nan
    # overall sensitivity of obs
    sensitivity_ens = read_data(data_path('study2/results/result_july/Ensemble-8obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_singleSMOUT.npy'))
    sensitivity_ens[:, 3, :, :, :][sensitivity_ens[:, 4, :, :, :] > 0.05] = np.nan
    overall_sen[0:2, :, :] = np.nanmean(sensitivity_ens[0:4, 3, 1:3, :, :],axis=0)  # obs

    # overall sensitivity of model
    sensitivity_ens = read_data(data_path('study2/results/result_april/TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_SM13modelOUT.npy'))
    sensitivity_ens[9, 3, :, :, :][sensitivity_ens[9, 4, :, :, :] > 0.05] = np.nan
    overall_sen[2:4, :, :] = sensitivity_ens[9, 3, 1:3, :, :]  # model
    # NaN masks by obs, model
    overall_sen[0, :, :][np.isnan(overall_sen[1, :, :])] = np.nan
    overall_sen[1, :, :][np.isnan(overall_sen[0, :, :])] = np.nan
    overall_sen[2, :, :][np.isnan(overall_sen[3, :, :])] = np.nan
    overall_sen[3, :, :][np.isnan(overall_sen[2, :, :])] = np.nan
    overall_sen[0:2, :, :][np.isnan(overall_sen[2:4, :, :])] = np.nan
    overall_sen[2:4, :, :][np.isnan(overall_sen[0:2, :, :])] = np.nan

    overall_sen[4:6,:,:] = overall_sen[2:4,:,:]-overall_sen[0:2,:,:] #diff

    for v in range(6):
        overall_sen[v, :, :][overall_sen[v, :, :] < np.round(np.nanpercentile(overall_sen[v, :, :], 5), 3)] = np.round(np.nanpercentile(overall_sen[v, :, :], 5), 3)
        overall_sen[v, :, :][overall_sen[v, :, :] > np.round(np.nanpercentile(overall_sen[v, :, :], 95), 3)] = np.round(np.nanpercentile(overall_sen[v, :, :], 95), 3)
        overall_sen[v, :, :][vegetation_cover < 5] = np.nan
        overall_sen[v, :, :][irrigation > 10] = np.nan

    print(np.nanpercentile(overall_sen[0, :, :],5),np.nanpercentile(overall_sen[0, :, :],95))
    print(np.nanpercentile(overall_sen[1, :, :],5),np.nanpercentile(overall_sen[1, :, :],95))
    print(np.nanpercentile(overall_sen[2, :, :],5),np.nanpercentile(overall_sen[2, :, :],95))
    print(np.nanpercentile(overall_sen[3, :, :],5),np.nanpercentile(overall_sen[3, :, :],95))
    print(np.nanpercentile(overall_sen[4, :, :],5),np.nanpercentile(overall_sen[4, :, :],95))
    print(np.nanpercentile(overall_sen[5, :, :],5),np.nanpercentile(overall_sen[5, :, :],95))

    # create figures
    fig = plt.figure(figsize=[8,5], dpi=300, tight_layout=True)
    gs   = gridspec.GridSpec(5, 3, height_ratios=[1,1,0.05,0.2,0.8], width_ratios=[1,1,0.1], hspace=0.1, wspace=0.02)
    gs01 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[6], width_ratios=[0.1,4,0.1])
    gs02 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[7], width_ratios=[0.1,4,0.1])
    gs03 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[12], width_ratios=[0.15,1,0.15,1,0.1])
    gs04 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[13], width_ratios=[0.15,1,0.15,1,0.1])

    cmap = plt.get_cmap('coolwarm')
    min = -0.03
    max = 0.03
    norm1 = colors.TwoSlopeNorm(vmin=min, vcenter=0, vmax=max)
    ticks1 = min, min / 2, 0, max / 2, max
    min = -0.005
    max = 0.005
    norm2 = colors.TwoSlopeNorm(vmin=min, vcenter=0, vmax=max)
    ticks2 = min, min / 2, 0, max / 2, max

    # parameters to make heatmaps
    Temp = read_data(data_path('study2/results/result_april/EAR5-land_6vtrend_lai0.5_t2m5_1982_2017_yearlymean.npy'))[:,3, :, :] - 273.15
    Temp = np.nanmean(Temp, axis=0)
    Aridity = read_data(data_path('study2/results/result_april/EAR5-land_aridity_1982_2017_mean.npy'))
    Temp[Temp < 5] = 5
    Temp[Temp > 31] = 31
    Aridity[Aridity < 0] = 0
    Aridity[Aridity > 4] = 4
    xtick_20 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.4, 2.8, 3.2, 3.6, 4]
    ytick_20 = np.linspace(5, 31, 21)
    xtick_4 = [0, 0.5, 1, 2, 4]
    ytick_4 =[5,11,17,23,31]
    heatmap_data_20x20, heatmap_data_4x4, num1 = heat_map_20(Aridity, Temp, overall_sen, xtick_20, ytick_20, xtick_4, ytick_4)  # combine heatmaps
    heatmap_data_corr,num2 = heat_map_corr(Aridity, Temp, overall_sen[0:2, :, :], overall_sen[2:4, :, :], xtick_4, ytick_4)  # combine heatmaps

    ########## draw obs, models
    ax00 = fig.add_subplot(gs[0])
    cs = graph(ax00, overall_sen[0, 30:300, :], norm=norm1, cmap=cmap)
    ax01 = plt.axes([0.15, 0.69, 0.06, 0.08])  # [left,bottom,width,height]
    heatmap_20(ax01, heatmap_data_20x20[0,:,:], norm=norm1, cmap=cmap)

    ax10 = fig.add_subplot(gs[1])
    cs = graph(ax10, overall_sen[1, 30:300, :], norm=norm2, cmap=cmap)
    ax11 = plt.axes([0.519, 0.69, 0.06, 0.08])  # [left,bottom,width,height]
    heatmap_20(ax11, heatmap_data_20x20[1,:,:], norm=norm2, cmap=cmap)

    ax20 = fig.add_subplot(gs[3])
    cs = graph(ax20, overall_sen[2, 30:300, :], norm=norm1, cmap=cmap)
    ax21 = plt.axes([0.15, 0.44, 0.06, 0.08])  # [left,bottom,width,height]
    heatmap_20(ax21, heatmap_data_20x20[2, :, :], norm=norm1, cmap=cmap)

    ax30 = fig.add_subplot(gs[4])
    cs = graph(ax30, overall_sen[3, 30:300, :], norm=norm2, cmap=cmap)
    ax31 = plt.axes([0.519, 0.44, 0.06, 0.08])  # [left,bottom,width,height]
    heatmap_20(ax31, heatmap_data_20x20[3, :, :], norm=norm2, cmap=cmap)

    ax00.set_title('Overall ' + tit[0] + unit)
    ax10.set_title('Overall ' + tit[1] + unit)

    ########### draw colorbar
    ax6 = fig.add_subplot(gs01[1])
    cbar = matplotlib.colorbar.ColorbarBase(ax6, norm=norm1, ticks=ticks1, cmap=cmap, extend='both',orientation='horizontal')
    ax7 = fig.add_subplot(gs02[1])
    cbar = matplotlib.colorbar.ColorbarBase(ax7, norm=norm2, ticks=ticks2, cmap=cmap, extend='both',orientation='horizontal')

    ########### draw diff
    cmap1 = plt.get_cmap('YlGn')
    ax41 = fig.add_subplot(gs03[1])
    ax41.set_ylabel('Temperature ($^\circ$C)')
    ax42 = fig.add_subplot(gs03[3])
    heatmap_corr(ax41, heatmap_data_4x4[4, :, :], [-0.03,0,0.03], norm=norm1, cmap=cmap)
    print(np.nanmin(heatmap_data_corr[0, :, :]),np.nanmax(heatmap_data_corr[0, :, :]))
    heatmap_corr(ax42, heatmap_data_corr[0, :, :], [-0.7,0,0.7], norm=colors.TwoSlopeNorm(vmin=-0.7, vcenter=0, vmax=0.7), cmap=cmap1)

    ax51 = fig.add_subplot(gs04[1])
    ax52 = fig.add_subplot(gs04[3])
    heatmap_corr(ax51, heatmap_data_4x4[5, :, :], [-0.005,0,0.005], norm=norm2, cmap=cmap)
    print(np.nanmin(heatmap_data_corr[1, :, :]),np.nanmax(heatmap_data_corr[1, :, :]))
    heatmap_corr(ax52, heatmap_data_corr[1, :, :], [-0.7,0,0.7], norm=colors.TwoSlopeNorm(vmin=-0.7, vcenter=0, vmax=0.7), cmap=cmap1)

    ##########
    ax6.text(-0.038, 2.6, '(A) obs')
    ax6.text(-0.038, 1.32, '(B) Model')
    ax6.text(-0.028, -0.36, '(C) Model-obs')
    ax6.text(0.007, -0.36, '(D) Corr(model,obs)')
    ax7.text(-0.0062, 0.43, '(E) obs')
    ax7.text(-0.0062, 0.22, '(F) Model')
    ax7.text(-0.0048, -0.06, '(G) Model-obs')
    ax7.text(0.001, -0.06, '(H) Corr(model,obs)')
    plt.savefig('/Net/Groups/BGI/scratch/wantong/study2/results/result_april/figure2/fig1.jpg', bbox_inches='tight')


    ######### plot a new figure which showing the numbers of grid cells in heatmap
    fig = plt.figure(figsize=[2,2], dpi=300, tight_layout=True)
    ax = fig.add_subplot(1,1,1)
    heatmap_num(num2[0,:,:], ax, np.ones((4,4)))
    plt.savefig('/Net/Groups/BGI/scratch/wantong/study2/results/result_april/figure2/figs2.jpg', bbox_inches='tight')

    print('end')



