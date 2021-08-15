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
import matplotlib.gridspec as gridspec

lat=360
lon=720
year1=1982
year2=2017

tit = [r'$\frac{\partial{LAI}}{\partial{SMsurf}}$',r'$\frac{\partial{LAI}}{\partial{SMroot}}$',r'$\frac{\partial{LAI}}{\partial{SMsurf}}$',r'$\frac{\partial{LAI}}{\partial{SMroot}}$']
unit1 = ' ($\mathregular{mm^{-1}}$ per 3 years)'
unit2 = ' ($\mathregular{mm^{-1}}$)'

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
plt.rcParams.update({'font.size': 6})
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

# def heatmap(num, ax, data, ytick, xtick, ticks, label, cbar_kw={}, **kwargs):
def heatmap(num, ax, data, ytick, xtick, **kwargs):
    im1 = ax.imshow(data, **kwargs, origin='lower')
    # cbar = ax.figure.colorbar(im1, ax=ax, ticks=ticks, **cbar_kw, extend='both')
    # cbar.set_ticklabels(label)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5)
    ax.set_xticklabels(xtick)
    ax.set_yticklabels(ytick)

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    # for y in range(4):
    #     for x in range(4):
    #         ax.text(x, y, num[y, x], ha="center", va="center", color="white")

    return im1

if __name__ == '__main__':
    log_string = 'heatmap :'

    # short-to-tall vegetation ratio
    fvc = read_data(data_path('Proj1VD/original_data/Landcover/VCF5KYR/vcf5kyr_v001/VCF_1982_to_2016_0Tree_1nonTree_yearly.npy'))
    grass_fraction_long_mean = np.nanmean(fvc[1, :, :, :], axis=0) / np.nanmean(fvc[0, :, :, :], axis=0)
    grass_fraction_long_mean[grass_fraction_long_mean < 0] = np.nan
    grass_fraction_long_mean[grass_fraction_long_mean > 100] = np.nan

    overall_sen = np.zeros((4, 360, 720)) * np.nan
    # overall sensitivity of obs
    sensitivity_ens = read_data(data_path('study2/results/result_july/Ensemble-8obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_singleSMOUT.npy'))
    sensitivity_ens[:, 3, :, :, :][sensitivity_ens[:, 4, :, :, :] > 0.05] = np.nan
    sensitivity_ens[:, 3, :, :, :][sensitivity_ens[:, 3, :, :, :] < 0] = np.nan
    overall_sen[0:2, :, :] = np.nanmean(sensitivity_ens[0:4, 3, 1:3, :, :], axis=0)  # obs
    # overall sensitivity of model
    sensitivity_ens = read_data(data_path('study2/results/result_april/TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_SM13modelOUT.npy'))
    sensitivity_ens[9, 3, :, :, :][sensitivity_ens[9, 4, :, :, :] > 0.05] = np.nan
    sensitivity_ens[9, 3, :, :, :][sensitivity_ens[9, 3, :, :, :] < 0] = np.nan
    overall_sen[2:4, :, :] = sensitivity_ens[9, 3, 1:3, :, :]  # model

    # NaN masks by obs, model
    overall_sen[0:2, :, :][np.isnan(overall_sen[2:4, :, :])] = np.nan
    overall_sen[2:4, :, :][np.isnan(overall_sen[0:2, :, :])] = np.nan

    # mask VCF<5%, irrigation>10%
    irrigation = read_data(data_path('study2/original_data/irrigation/gmia_v5_aei_pct_360_720.npy'))
    irrigation = np.repeat(irrigation[np.newaxis, :, :], 4, axis=0)
    overall_sen[irrigation > 10] = np.nan
    vegetation_cover = np.nanmean(fvc[0, :, :, :] + fvc[1, :, :, :], axis=0)
    vegetation_cover = np.repeat(vegetation_cover[np.newaxis, :, :], 4, axis=0)
    overall_sen[vegetation_cover < 0.05] = np.nan

    xticks = [[0,0.007,0.014,0.021,0.032],[0,0.01,0.02,0.03,0.045]]
    xticks_new = [[0,0.007,0.014,0.021,0.028],[0,0.01,0.02,0.03,0.04]]

    yticks = [[0,0.001,0.002,0.003,0.004],[0,0.005,0.01,0.015,0.022]]
    yticks_new = [[0,0.001,0.002,0.003,0.004],[0,0.005,0.01,0.015,0.02]]

    fig = plt.figure(figsize=(4,1.7), dpi=300, tight_layout=True)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,0.05], wspace=0)
    gs00 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[2], height_ratios=[0.4,1,0.4])

    for v in range(2):
        if v==0:
            ax = fig.add_subplot(gs[0])
        elif v==1:
            ax = fig.add_subplot(gs[1])

        arid_group = overall_sen[v*2, :, :]
        t2m_group = overall_sen[v*2+1, :, :]
        arid_group[arid_group > np.nanpercentile(arid_group, 95)] = np.nanpercentile(arid_group, 95)
        t2m_group[t2m_group > np.nanpercentile(t2m_group, 95)] = np.nanpercentile(t2m_group, 95)

        ytick = yticks[v]
        ytick_new = yticks_new[v]

        xtick = xticks[v]
        xtick_new = xticks_new[v]
        # ytick = np.round(np.linspace(0, np.nanmax(t2m_group), 5), 3)
        # xtick = np.round(np.linspace(0, np.nanmax(arid_group), 5), 3)
        # ytick_new = ytick
        # xtick_new = xtick

        heatmap_data,num = heat_map(arid_group, t2m_group, grass_fraction_long_mean, xtick, ytick)
        max = np.nanmax(heatmap_data)
        min = np.nanmin(heatmap_data)
        print(min,max)
        print(heatmap_data)

        norm = colors.Normalize(vmin=0, vmax=6)

        heatmap(num, ax, heatmap_data, ytick_new, xtick_new, norm=norm, cmap=plt.get_cmap('YlGn'))
        if v==0:
            ax.set_title('(A) obs')
            ax.set_ylabel('Overall ' + tit[1] + unit2)

        if v==1:
            ax.set_title('(B) Model')

        ax.set_xlabel('Overall ' + tit[0] + unit2)

    # draw colorbar
    ax01 = fig.add_subplot(gs00[1])
    norm = colors.Normalize(vmin=0, vmax=6)
    ticks = [0,1.5,3,4.5,6]
    matplotlib.colorbar.ColorbarBase(ax01, ticks=ticks, norm=norm, extend='both', cmap=plt.get_cmap('YlGn'))
    ax01.set_yticklabels(['<1.5',1.5,3,4.5,'>4.5'])

    ax01.text(0, 8, 'Short-to-tall\nvegetation ratio')

    fig.savefig(data_path('study2/results/result_april/figure2/fig3.jpg'),bbox_inches='tight')



    print('end')

