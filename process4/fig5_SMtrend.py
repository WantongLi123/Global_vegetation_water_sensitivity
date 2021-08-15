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
    #         ax.text(x, y, num[y, x], ha="center", va="center", color="black")

    return im1

if __name__ == '__main__':
    log_string = 'heatmap :'

    # plot trends of sensitivities binned by 5x5 arid x T2M
    Slope = np.zeros((4,360,720)) * np.nan
    P_value = np.zeros((4,360,720)) * np.nan
    Slope[0:2,:,:] = read_data('/Net/Groups/BGI/scratch/wantong/study2/results/result_july/ELAI_SM12_ERA5-land_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy')[0,:,:,:]
    P_value[0:2,:,:] = read_data('/Net/Groups/BGI/scratch/wantong/study2/results/result_july/ELAI_SM12_ERA5-land_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy')[1,:,:,:]
    Slope[2:4,:,:] = read_data('/Net/Groups/BGI/scratch/wantong/study2/results/result_july/TRENDY_S3_SM12_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy')[0, :, :, :]
    P_value[2:4,:,:] = read_data('/Net/Groups/BGI/scratch/wantong/study2/results/result_july/TRENDY_S3_SM12_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy')[1, :, :, :]
    # Slope[P_value>0.1]=np.nan
    # Slope[P_value<0]=np.nan

    # mask VCF<5%, irrigation>10%
    irrigation = read_data(data_path('study2/original_data/irrigation/gmia_v5_aei_pct_360_720.npy'))
    irrigation = np.repeat(irrigation[np.newaxis, :, :], 4, axis=0)
    Slope[irrigation > 10] = np.nan
    fvc = read_data(data_path('Proj1VD/original_data/Landcover/VCF5KYR/vcf5kyr_v001/VCF_1982_to_2016_0Tree_1nonTree_yearly.npy'))
    vegetation_cover = np.nanmean(fvc[0,:,:,:]+fvc[1,:,:,:], axis=0)
    vegetation_cover = np.repeat(vegetation_cover[np.newaxis, :, :], 4, axis=0)
    Slope[vegetation_cover < 0.05] = np.nan

    v6_slope = read_data(data_path('study2/results/result_july/ESM-obs-model_SM12_1982_2017_growseason_yearlymean_mkSlope.npy')) # from index 0 to 3: SMsurf_obs, SMroot_obs, SMsurf_model, SMroot_model
    v6_slope = v6_slope * 3

    # SM/T Trend - overall sensitivity
    over_sensi = np.zeros((4,360,720)) * np.nan
    over_p = np.zeros((4,360,720)) * np.nan
    over_sensi[0:2,:,:] = read_data(data_path('study2/results/result_july/Ensemble-8obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_singleSMOUT.npy'))[7, 3, 1:3, :, :]
    over_p[0:2,:,:] = read_data(data_path('study2/results/result_july/Ensemble-8obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_singleSMOUT.npy'))[7, 4, 1:3, :, :]
    over_sensi[2:4, :, :] = read_data(data_path('study2/results/result_april/TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_SM13modelOUT.npy'))[9, 3, 1:3, :, :]
    over_p[2:4,:,:] = read_data(data_path('study2/results/result_april/TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_SM13modelOUT.npy'))[9, 4, 1:3, :, :]
    over_sensi[over_sensi <= 0] = np.nan
    over_sensi[over_p >= 0.01] = np.nan
    over_sensi[2:4, :, :][np.isnan(over_sensi[0:2, :, :])] = np.nan
    over_sensi[0:2, :, :][np.isnan(over_sensi[2:4, :, :])] = np.nan

    xlabel = ['Trends of SMsurf (mm per 3 years)', 'Trends of SMroot (mm per 3 years)','Trends of SMsurf (mm per 3 years)', 'Trends of SMroot (mm per 3 years)']
    ylabel = ['Trends of SMsurf', 'Trends of SMroot','Trends of SMsurf', 'Trends of SMroot']
    xticks = [[-0.3, -0.1, 0, 0.1, 0.3], [-3, -1, 0, 1, 3],[-0.2, -0.05, 0, 0.05, 0.2], [-2, -0.5, 0, 0.5, 2]]
    xticks_new = [[-0.2, -0.1, 0, 0.1, 0.2], [-2, -1, 0, 1, 2],[-0.1, -0.05, 0, 0.05, 0.1], [-1, -0.5, 0, 0.5, 1]]

    yticks = [[0, 0.007, 0.014, 0.021, 0.028], [0, 0.001, 0.002, 0.003, 0.004],[0, 0.01, 0.02, 0.03, 0.04], [0, 0.006, 0.012, 0.018, 0.024]]
    yticks_new = yticks

    fig = plt.figure(figsize=(4.5,3.2), dpi=300, tight_layout=True)
    gs = gridspec.GridSpec(1, 5, width_ratios=[1,0.05,0.5,1,0.05], wspace=0)
    gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.3)
    gs01 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1], height_ratios=[0.4,1,0.4])
    gs10 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[3], hspace=0.3)
    gs11 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[4], height_ratios=[0.4,1,0.4])

    for v in range(4):
        if v==0:
            ax = fig.add_subplot(gs00[0])
        elif v==1:
            ax = fig.add_subplot(gs10[0])
        elif v==2:
            ax = fig.add_subplot(gs00[1])
        else:
            ax = fig.add_subplot(gs10[1])
        arid_group = v6_slope[v, :, :]
        t2m_group = over_sensi[v, :, :]
        t2m_group[t2m_group > np.nanpercentile(t2m_group, 95)] = np.nanpercentile(t2m_group, 95)
        ytick = yticks[v]
        ytick_new = yticks_new[v]
        # ytick = np.round(np.linspace(0, np.nanmax(t2m_group), 5), 3)
        # ytick_new = ytick

        arid_group[arid_group < np.nanpercentile(arid_group, 5)] = np.nanpercentile(arid_group, 5)
        arid_group[arid_group > np.nanpercentile(arid_group, 95)] = np.nanpercentile(arid_group, 95)
        xtick = xticks[v]
        xtick_new = xticks_new[v]
        # xtick = np.nanpercentile(arid_group, 0), np.nanpercentile(arid_group,0) / 2, 0, np.nanpercentile(arid_group, 100) / 2, np.nanpercentile(arid_group, 100)
        # xtick_new=np.round(xtick,3)

        heatmap_data,num = heat_map(arid_group, t2m_group, Slope[v, :, :], xtick, ytick)
        max = np.nanmax(heatmap_data)
        min = np.nanmin(heatmap_data)
        print(min,max)
        print(heatmap_data)

        if v==0 or v==2:
            norm = colors.TwoSlopeNorm(vmin=-0.0006, vcenter=0, vmax=0.0006)
        if v == 1 or v == 3:
            norm = colors.TwoSlopeNorm(vmin=-0.0001, vcenter=0, vmax=0.0001)

        # if v==0 or v==2:
        #     min=-0.0006
        #     max=0.0006
        # if v==1 or v==3:
        #     min = -0.0001
        #     max = 0.0001
        # if max>0 and min<0:
        #     norm = colors.TwoSlopeNorm(vmin=min, vcenter=0, vmax=max)
        #     ticks = min, min / 2, 0, max / 2, max
        #     label = np.round(ticks, 4)
        #     cmap = 'coolwarm'
        # elif max<=0:
        #     norm = colors.Normalize(vmin=min, vmax=max)
        #     ticks = min, min*3/4, min / 2, min / 4, 0
        #     label = np.round(ticks, 4)
        #     cmap = truncate_colormap(plt.get_cmap('coolwarm'), 0, 0.5)
        # elif min>=0:
        #     norm = colors.Normalize(vmin=min, vmax=max)
        #     ticks = 0, max/4, max / 2, max * 3 / 4, max
        #     label = np.round(ticks, 4)
        #     cmap = truncate_colormap(plt.get_cmap('coolwarm'), 0.5, 1)

        # heatmap(num, ax, heatmap_data, ytick_new, xtick_new, ticks, label, cmap=cmap, norm=norm, cbar_kw=dict(shrink=0.8))
        heatmap(num, ax, heatmap_data, ytick_new, xtick_new, norm=norm, cmap=plt.get_cmap('coolwarm'))
        if v==0 or v==1:
            ax.set_title('Trends of ' + tit[v] + unit1)
        if v==2 or v==3:
            ax.set_xlabel(xlabel[v])
        if v==0 or v==2:
            ax.set_ylabel('Overall ' + tit[v] + unit2)

    # draw colorbar
    ax01 = fig.add_subplot(gs01[1])
    norm = colors.Normalize(vmin=-0.0006, vmax=0.0006)
    ticks = [-0.0006, -0.0003, 0, 0.0003, 0.0006]
    matplotlib.colorbar.ColorbarBase(ax01, ticks=ticks, norm=norm, extend='both', cmap=plt.get_cmap('coolwarm'))
    ax01.set_yticklabels(['-6e-04', '-3e-04', '0', '3e-04', '6e-04'])

    ax11 = fig.add_subplot(gs11[1])
    norm = colors.Normalize(vmin=-0.0001, vmax=0.0001)
    ticks = [-0.0001,-0.00005,0,0.00005,0.0001]
    matplotlib.colorbar.ColorbarBase(ax11, ticks=ticks, norm=norm, extend='both', cmap=plt.get_cmap('coolwarm'))
    ax11.set_yticklabels(['-1e-04', '-0.5e-04', '0', '0.5e-04', '1e-04'])

    ax01.text(-0.03, 0.0015, '(a) obs')
    ax01.text(0.008, 0.0015, '(c) obs')
    ax01.text(-0.03, -0.0001, '(b) Model')
    ax01.text(0.008, -0.0001, '(d) Model')

    fig.savefig(data_path('study2/results/result_april/figure2/fig4.jpg'),bbox_inches='tight')



    print('end')

