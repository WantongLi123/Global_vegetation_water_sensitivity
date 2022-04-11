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
import imageio
import pymannkendall as mk
from matplotlib import rc
import matplotlib.gridspec as gridspec

lat=360
lon=720
year1=1982
year2=2017

tit = [r'$\frac{\partial{LAI}}{\partial{SMnear}}$',r'$\frac{\partial{LAI}}{\partial{SMsub}}$',r'$\frac{\partial{LAI}}{\partial{SMnear}}$',r'$\frac{\partial{LAI}}{\partial{SMsub}}$']
unit1 = ' ($\mathregular{mm^{-1}}$ per 3 years)'
unit2 = ' ($\mathregular{mm^{-1}}$)'
obs_model = ['Obs', '(a) Obs', 'Model', '(b) Model']

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
plt.rcParams.update({'font.size': 5})
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

def yearlytotal(year,lai, t2m, variable):
    yearly_total = np.zeros((year, 360, 720)) * np.nan
    variable[lai<=0.5] = np.nan
    variable[t2m<=278.15] = np.nan

    for row in range(360):
        for col in range(720):
            if np.isnan(variable[:, row, col]).all():
                yearly_total[:, row, col] = np.nan
            else:
                for y in range(year):
                    yearly_total[y, row, col] = np.nansum(variable[y*12:(y*12+12), row, col])
    return (yearly_total)

def slope_yearly(variable):
    slope = np.zeros((360, 720)) * np.nan
    for row in range(360):
        for col in range(720):
                if np.isnan(variable[:, row, col]).all() or np.all(variable[:, row, col]==0):
                    slope[row, col] = np.nan
                else:
                    y = variable[:, row, col]
                    y = y[~np.isnan(variable[:, row, col])]
                    if len(y)>5:
                        result = mk.original_test(y)  # mk2
                        slope[row, col] = result.slope * 3 # trends per 3 years
    return (slope)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def heat_map(arid, tem, c, x_axis,y_axis):
    data = np.zeros((4, 4))
    num = np.zeros((4, 4), dtype=np.int)
    for y in range(4):
        for x in range(4):
            with np.errstate(invalid='ignore'):
                mask = c[np.where((arid >= x_axis[x]) & (arid <= x_axis[x + 1]) &
                                  (tem >= y_axis[y]) & (tem <= y_axis[y + 1]))]
                num[y,x] = np.sum(~np.isnan(mask))
                if np.sum(~np.isnan(mask))>=20:
                    data[y, x] = np.nanmedian(mask)

    return(data,num)

def heatmap(num, ax, data, ytick, xtick, **kwargs):
    im1 = ax.imshow(data, origin='lower', **kwargs)
    # cbar = ax.figure.colorbar(im1, ax=ax, ticks=ticks, **cbar_kw, extend='both')
    # cbar.set_ticklabels(label)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5)
    ax.set_xticklabels(xtick)
    ax.set_yticklabels(ytick)

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    for y in range(4):
        for x in range(4):
            ax.text(x, y, num[y, x], ha="center", va="center", color="white")

    return im1

if __name__ == '__main__':
    log_string = 'heatmap :'

    # plot trends of sensitivities binned by 5x5 arid x T2M
    Slope = np.zeros((4,360,720)) * np.nan
    P_value = np.zeros((4,360,720)) * np.nan
    Slope[0:2,:,:] = read_data('ELAI_SM12_ERA5-land_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy')[0,:,:,:] # obtain data from fig3.py
    P_value[0:2,:,:] = read_data('ELAI_SM12_ERA5-land_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy')[1,:,:,:]# obtain data from fig3.py
    Slope[2:4,:,:] = read_data('TRENDY_S3_SM12_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy')[0, :, :, :]# obtain data from fig3.py
    P_value[2:4,:,:] = read_data('TRENDY_S3_SM12_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy')[1, :, :, :]# obtain data from fig3.py

    # mask VCF<5%, irrigation>10%
    irrigation = read_data(data_path('gmia_v5_aei_pct_360_720.npy'))
    irrigation = np.repeat(irrigation[np.newaxis, :, :], 4, axis=0)
    Slope[irrigation > 10] = np.nan
    fvc = read_data(data_path('VCF_1982_to_2016_0Tree_1nonTree_yearly.npy'))
    vegetation_cover = np.nanmean(fvc[0,:,:,:]+fvc[1,:,:,:], axis=0)
    vegetation_cover = np.repeat(vegetation_cover[np.newaxis, :, :], 4, axis=0)
    Slope[vegetation_cover < 0.05] = np.nan

    # calculate precipitation trends
    T2M = read_data(data_path('CRUJRAtmp_444_0.5.npy'))
    TP = read_data(data_path('CRUJRApre_444_0.5.npy'))
    EnLAI = read_data(data_path('study2/original_data/LAI/EnLAI_5mean_1982_2018_monthly_0.5.npy'))
    TP_total = yearlytotal(year,EnLAI[0:432, :, :], T2M[0:432, :, :], TP[0:432, :, :])
    TP_slope = slope_yearly(TP_total)
    
    # Precipitation Trend as x-axis; overall sensitivity as y-axis
    over_sensi = np.zeros((4,360,720)) * np.nan
    over_p = np.zeros((4,360,720)) * np.nan
    over_sensi[0:2,:,:] = read_data(data_path('Ensemble-6obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear.npy'))[5, 3, 1:3, :, :]
    over_p[0:2,:,:] = read_data(data_path('Ensemble-6obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear.npy'))[5, 4, 1:3, :, :]
    over_sensi[2:4, :, :] = read_data(data_path('study2/results/result_april/TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_SM13modelOUT.npy'))[9, 3, 1:3, :, :]
    over_p[2:4,:,:] = read_data(data_path('study2/results/result_april/TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_SM13modelOUT.npy'))[9, 4, 1:3, :, :]
    over_sensi[over_sensi <= 0] = np.nan
    over_sensi[over_p >= 0.01] = np.nan
    over_sensi[2:4, :, :][np.isnan(over_sensi[0:2, :, :])] = np.nan
    over_sensi[0:2, :, :][np.isnan(over_sensi[2:4, :, :])] = np.nan

    xticks = [-16, -5, 0, 5, 16]
    xticks_new = [-10, -5, 0, 5, 10]

    yticks = [[0, 0.004, 0.008, 0.012, 0.016], [0, 0.0005, 0.001, 0.0015, 0.002],[0, 0.01, 0.02, 0.03, 0.04], [0, 0.006, 0.012, 0.018, 0.024]]
    yticks_new = yticks

    # figure frame
    fig = plt.figure(figsize=(4,2), dpi=300, tight_layout=True)
    gs = gridspec.GridSpec(1, 5, width_ratios=[1,0.3,1,0.05,0.06], wspace=0)
    gs00 = gridspec.GridSpecFromSubplotSpec(3, 1, height_ratios=[0.3,1,0.3], subplot_spec=gs[4], wspace=0)


    for v in [1,3]: # change here to [0,2] for near-surface soil moisture
        if v==0 or v==1:
            ax = fig.add_subplot(gs[0])
        elif v==2 or v==3:
            ax = fig.add_subplot(gs[2])

        arid_group = TP_slope
        t2m_group = over_sensi[v, :, :]
        t2m_group[t2m_group > np.nanpercentile(t2m_group, 95)] = np.nanpercentile(t2m_group, 95)
        ytick = yticks[v]
        ytick_new = yticks_new[v]

        arid_group[arid_group < np.nanpercentile(arid_group, 5)] = np.nanpercentile(arid_group, 5)
        arid_group[arid_group > np.nanpercentile(arid_group, 95)] = np.nanpercentile(arid_group, 95)
        xtick = xticks
        xtick_new = xticks_new

        heatmap_data,num = heat_map(arid_group, t2m_group, Slope[v, :, :], xtick, ytick)
        max = np.nanmax(heatmap_data)
        min = np.nanmin(heatmap_data)
        print(min,max)
        print(heatmap_data)

        norm = colors.TwoSlopeNorm(vmin=-0.00004, vcenter=0, vmax=0.00004)
        if v == 1 or v == 3:
            norm = colors.TwoSlopeNorm(vmin=-0.00004, vcenter=0, vmax=0.00004)

        heatmap(num, ax, heatmap_data, ytick_new, xtick_new, norm=norm, cmap=plt.get_cmap('coolwarm'))
        ax.set_title(obs_model[v])
        ax.set_xlabel('Total precipitation trends (mm per 3 years)')

        if v == 0:
            ax.set_ylabel('Overall ' + tit[v] + unit2)
        if v == 1:
            ax.set_ylabel('Overall ' + tit[v] + unit2)

    # draw colorbar
    ax01 = fig.add_subplot(gs00[1])
    norm = colors.Normalize(vmin=-0.00004, vmax=0.00004)
    ticks = [-0.00004,-0.00002,0,0.00002,0.00004]
    cbar = matplotlib.colorbar.ColorbarBase(ax01, ticks=ticks, norm=norm, extend='both', cmap=plt.get_cmap('coolwarm'))
    eee='$\mathregular{x10^{-5}}$'
    ax01.set_yticklabels(['-4.0'+eee, '-2.0'+eee, '0', '2.0'+eee, '4.0'+eee])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Trends of ' + tit[1] + unit1, rotation=270)

    fig.savefig(data_path('fig4.jpg'),bbox_inches='tight')



    print('end')

