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

cmap_color = cmaps.ncl_default  # 如需反转colorbar，其方法是cmap_color=cmap_color.reversed()
import imageio
import pymannkendall as mk
from matplotlib import rc
import matplotlib.gridspec as gridspec

lat = 360
lon = 720
year1 = 1982
year2 = 2017

tit = [r'$\frac{\partial{LAI}}{\partial{SMsurf}}$', r'$\frac{\partial{LAI}}{\partial{SMroot}}$',
       r'$\frac{\partial{LAI}}{\partial{SMsurf}}$', r'$\frac{\partial{LAI}}{\partial{SMroot}}$']
unit1 = ' ($\mathregular{mm^{-1}}$ per 3 years)'
unit2 = ' ($\mathregular{mm^{-1}}$)'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
plt.rcParams.update({'font.size': 5})
plt.rcParams['axes.unicode_minus'] = False  # correct the sign of figure'' labels when there is a negative value


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


def heat_map(arid, tem, c, x_axis, y_axis):
    data = np.zeros((4, 4))
    num = np.zeros((4, 4), dtype=np.int)
    for y in range(4):
        for x in range(4):
            with np.errstate(invalid='ignore'):
                mask = c[np.where((arid >= x_axis[x]) & (arid <= x_axis[x + 1]) &
                                  (tem >= y_axis[y]) & (tem <= y_axis[y + 1]))]
                num[y, x] = np.sum(~np.isnan(mask))
                if np.sum(~np.isnan(mask)) >= 20:
                    data[y, x] = np.nanmedian(mask)

    return (data, num)


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
    for y in range(4):
        for x in range(4):
            ax.text(x, y, num[y, x], ha="center", va="center", color="black")

    return im1


if __name__ == '__main__':
    log_string = 'heatmap :'

    # plot trends of sensitivities binned by 5x5 arid x T2M
    Slope = np.zeros((4, 360, 720)) * np.nan
    P_value = np.zeros((4, 360, 720)) * np.nan
    Slope[0:2, :, :] = read_data(
        '/Net/Groups/BGI/scratch/wantong/study2/results/result_july/ELAI_SM12_ERA5-land_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy')[
                       0, :, :, :]
    P_value[0:2, :, :] = read_data(
        '/Net/Groups/BGI/scratch/wantong/study2/results/result_july/ELAI_SM12_ERA5-land_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy')[
                         1, :, :, :]
    Slope[2:4, :, :] = read_data(
        '/Net/Groups/BGI/scratch/wantong/study2/results/result_july/TRENDY_S3_SM12_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy')[
                       0, :, :, :]
    P_value[2:4, :, :] = read_data(
        '/Net/Groups/BGI/scratch/wantong/study2/results/result_july/TRENDY_S3_SM12_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy')[
                         1, :, :, :]
    # Slope[P_value>0.1]=np.nan
    # Slope[P_value<0]=np.nan

    # mask VCF<5%, irrigation>10%
    irrigation = read_data(data_path('study2/original_data/irrigation/gmia_v5_aei_pct_360_720.npy'))
    irrigation = np.repeat(irrigation[np.newaxis, :, :], 4, axis=0)
    Slope[irrigation > 10] = np.nan
    fvc = read_data(
        data_path('Proj1VD/original_data/Landcover/VCF5KYR/vcf5kyr_v001/VCF_1982_to_2016_0Tree_1nonTree_yearly.npy'))
    vegetation_cover = np.nanmean(fvc[0, :, :, :] + fvc[1, :, :, :], axis=0)
    vegetation_cover = np.repeat(vegetation_cover[np.newaxis, :, :], 4, axis=0)
    Slope[vegetation_cover < 0.05] = np.nan

    multiTrends_obs = read_data(data_path('study2/results/result_july/multi_trends_13vari_sfig5_obs.npy'))
    TP_slope = multiTrends_obs[8, :, :] * 3
    T2M_slope = multiTrends_obs[2, :, :] * 3

    multiTrends_model = read_data(data_path('study2/results/result_july/multi_trends_13vari_sfig5_model.npy'))
    SMroot_model_slope = multiTrends_obs[5, :, :] * 3

    # short-to-tall vegetation ratio
    grass_fraction_long_mean = np.nanmean(fvc[1, :, :, :], axis=0) / np.nanmean(fvc[0, :, :, :], axis=0)
    grass_fraction_long_mean[grass_fraction_long_mean < 0] = np.nan
    grass_fraction_long_mean[grass_fraction_long_mean > 100] = np.nan


    # SM/T Trend - overall sensitivity
    over_sensi = np.zeros((4, 360, 720)) * np.nan
    over_p = np.zeros((4, 360, 720)) * np.nan
    over_sensi[0:2, :, :] = read_data(data_path(
        'study2/results/result_july/Ensemble-8obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_singleSMOUT.npy'))[
                            7, 3, 1:3, :, :]
    over_p[0:2, :, :] = read_data(data_path(
        'study2/results/result_july/Ensemble-8obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_singleSMOUT.npy'))[
                        7, 4, 1:3, :, :]
    over_sensi[2:4, :, :] = read_data(data_path(
        'study2/results/result_april/TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_SM13modelOUT.npy'))[
                            9, 3, 1:3, :, :]
    over_p[2:4, :, :] = read_data(data_path(
        'study2/results/result_april/TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_SM13modelOUT.npy'))[
                        9, 4, 1:3, :, :]
    over_sensi[over_sensi <= 0] = np.nan
    over_sensi[over_p >= 0.01] = np.nan
    over_sensi[2:4, :, :][np.isnan(over_sensi[0:2, :, :])] = np.nan
    over_sensi[0:2, :, :][np.isnan(over_sensi[2:4, :, :])] = np.nan

    fig = plt.figure(figsize=(4.5, 3.2), dpi=300, tight_layout=True)
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 0.05, 0.5, 1, 0.05], wspace=0)
    gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.3)
    gs01 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1], height_ratios=[0.4, 1, 0.4])
    gs10 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[3], hspace=0.3)
    gs11 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[4], height_ratios=[0.4, 1, 0.4])

    for v in range(4):
        if v == 0:
            ax = fig.add_subplot(gs00[0])
        elif v == 1:
            ax = fig.add_subplot(gs10[0])
        elif v == 2:
            ax = fig.add_subplot(gs00[1])
        else:
            ax = fig.add_subplot(gs10[1])

        if v == 0:
            arid_group = grass_fraction_long_mean
            t2m_group = over_sensi[0, :, :]
            t2m_group[t2m_group > np.nanpercentile(t2m_group, 95)] = np.nanpercentile(t2m_group, 95)
            ytick = [0, 0.007, 0.014, 0.021, 0.028]
            ytick_new = [0, 0.007, 0.014, 0.021, 0.028]
            xtick = [0,1.5,3,4.5,100]
            xtick_new = ['<1',1.5,3,4.5,'>4.5']
            ax.set_xlabel('Short-to-tall vegetation ratio')
            ax.set_ylabel('Overall ' + tit[0] + unit2)
        if v==1:
            arid_group = TP_slope
            t2m_group = over_sensi[1, :, :]
            t2m_group[t2m_group > np.nanpercentile(t2m_group, 95)] = np.nanpercentile(t2m_group, 95)
            ytick = [0, 0.001, 0.002, 0.003, 0.004]
            ytick_new = [0, 0.001, 0.002, 0.003, 0.004]
            arid_group[arid_group < np.nanpercentile(arid_group, 5)] = np.nanpercentile(arid_group, 5)
            arid_group[arid_group > np.nanpercentile(arid_group, 95)] = np.nanpercentile(arid_group, 95)
            xtick = [-16, -5, 0, 5, 16]
            xtick_new = [-10, -5, 0, 5, 10]
            ax.set_xlabel('Total precipitation trends (mm per 3 years)')
            ax.set_ylabel('Overall ' + tit[1] + unit2)
        if v==2:
            arid_group = T2M_slope
            t2m_group = grass_fraction_long_mean
            ytick = [0,1.5,3,4.5,100]
            ytick_new = [0, 0.001, 0.002, 0.003, 0.004]
            arid_group[arid_group < np.nanpercentile(arid_group, 5)] = np.nanpercentile(arid_group, 5)
            arid_group[arid_group > np.nanpercentile(arid_group, 95)] = np.nanpercentile(arid_group, 95)
            xtick = np.nanpercentile(arid_group, 0), np.nanpercentile(arid_group, 0) / 2, 0, np.nanpercentile(
                arid_group, 100) / 2, np.nanpercentile(arid_group, 100)
            xtick_new = np.round(xtick, 3)
            ax.set_xlabel('Mean temperature trends (˚C per 3 years)')
            ax.set_ylabel('Short-to-tall vegetation ratio')
        if v == 3:
            arid_group = SMroot_model_slope
            t2m_group = grass_fraction_long_mean
            ytick = [0,1.5,3,4.5,100]
            ytick_new = ['<1',1.5,3,4.5,'>4.5']
            arid_group[arid_group < np.nanpercentile(arid_group, 5)] = np.nanpercentile(arid_group, 5)
            arid_group[arid_group > np.nanpercentile(arid_group, 95)] = np.nanpercentile(arid_group, 95)
            xtick = [-2, -0.5, 0, 0.5, 2]
            xtick_new = [-1, -0.5, 0, 0.5, 1]
            ax.set_xlabel('Mean SMroot trends (mm per 3 years)')
            ax.set_ylabel('Short-to-tall vegetation ratio')

        heatmap_data, num = heat_map(arid_group, t2m_group, Slope[v, :, :], xtick, ytick)
        max = np.nanmax(heatmap_data)
        min = np.nanmin(heatmap_data)
        print(min, max)
        print(heatmap_data)

        if v == 0 or v == 2:
            norm = colors.TwoSlopeNorm(vmin=-0.0005, vcenter=0, vmax=0.0005)
        if v == 1 or v == 3:
            norm = colors.TwoSlopeNorm(vmin=-0.0001, vcenter=0, vmax=0.0001)

        # heatmap(num, ax, heatmap_data, ytick_new, xtick_new, ticks, label, cmap=cmap, norm=norm, cbar_kw=dict(shrink=0.8))
        heatmap(num, ax, heatmap_data, ytick_new, xtick_new, norm=norm, cmap=plt.get_cmap('coolwarm'))
        if v == 0 or v == 1:
            ax.set_title('Trends of ' + tit[v] + unit1)
        

    # draw colorbar
    ax01 = fig.add_subplot(gs01[1])
    norm = colors.Normalize(vmin=-0.0006, vmax=0.0006)
    ticks = [-0.0005, -0.00025, 0, 0.00025, 0.0005]
    matplotlib.colorbar.ColorbarBase(ax01, ticks=ticks, norm=norm, extend='both', cmap=plt.get_cmap('coolwarm'))
    ax01.set_yticklabels(['-5e-04', '-2.5e-04', '0', '2.5e-04', '5e-04'])

    ax11 = fig.add_subplot(gs11[1])
    norm = colors.Normalize(vmin=-0.0001, vmax=0.0001)
    ticks = [-0.0001, -0.00005, 0, 0.00005, 0.0001]
    matplotlib.colorbar.ColorbarBase(ax11, ticks=ticks, norm=norm, extend='both', cmap=plt.get_cmap('coolwarm'))
    ax11.set_yticklabels(['-1e-04', '-0.5e-04', '0', '0.5e-04', '1e-04'])

    ax01.text(-0.03, 0.0015, '(a) obs')
    ax01.text(0.008, 0.0015, '(c) obs')
    ax01.text(-0.03, -0.0001, '(b) Model')
    ax01.text(0.008, -0.0001, '(d) Model')

    fig.savefig(data_path('study2/results/result_april/figure2/fig4_first_two_variable.jpg'), bbox_inches='tight')

    print('end')

