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

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
plt.rcParams.update({'font.size': 8})
plt.rcParams['axes.unicode_minus']=False

tit = [r'$\frac{\partial{LAI}}{\partial{SMsurf}}$',r'$\frac{\partial{LAI}}{\partial{SMroot}}$']
unit = ' ($\mathregular{mm^{-1}}$)'
x_label = ['Growing-season SMsurf (mm)','Growing-season SMroot (mm)']
model = ['ISAM','LPJ-GUESS','LPX-Bern','VISIT','CABLE-POP','CLM5.0','JSBACH','JULES','ORCHIDEE-CNP','Ensemble model mean']

def data_path(filename):
    file_path = "{path}/{filename}".format(
        path="/Net/Groups/BGI/scratch/wantong",
        filename=filename
    )
    return file_path

def read_data(path):
    data = np.load(path)
    # print(log_string, path, 'read data')
    return data

def graph(ax, target, **kwargs):
    cs = ax.pcolor(target[::-1,:], **kwargs)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    return(cs)

def heatmap_20(ax, data, **kwargs):
    im1 = ax.pcolor(data, **kwargs)

    ax.set_xticks([0,10,20])
    ax.set_yticks([0,10,20])
    ax.set_xticklabels([0, 1, 4])
    ax.set_yticklabels([5,15,30])

    ax.set_xlabel('Aridity')
    ax.set_ylabel('Temperature ($^\circ$C)')

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    return(im1)

def interquantile_plot(ax, n, x_new, x, y, label, color):
    y1 = np.zeros(n) * np.nan
    y2 = np.zeros(n) * np.nan
    y3 = np.zeros(n) * np.nan
    number = np.zeros(n) * np.nan
    for i in range(n - 1):
        y1[i] = np.nanpercentile(y[np.where((x <= x_new[i + 1]) & (x >= x_new[i]))], 50)
        y2[i] = np.nanpercentile(y[np.where((x <= x_new[i + 1]) & (x >= x_new[i]))], 25)
        y3[i] = np.nanpercentile(y[np.where((x <= x_new[i + 1]) & (x >= x_new[i]))], 75)
        number[i] = np.sum(~np.isnan(y[np.where((x <= x_new[i + 1]) & (x >= x_new[i]))]))
    ax.plot(x_new, y1, '-', color=color, label = label, linewidth=1)
    ax.fill_between(x_new, y2, y3, facecolor=color, alpha=0.2)
    ax1 = ax.twinx()
    ax1.bar(x_new,number, width=(x_new[1]-x_new[0])*3/4, color=color, alpha=0.2)
    ax1.set_ylim([0, 15000])
    ax1.get_yaxis().set_ticks([])
    return(ax,ax1)

def interquantile_plot1(ax, n, x_new, x, y, color, width, legend, linestyle):
    y1 = np.zeros(n) * np.nan
    for i in range(n - 1):
        y1[i] = np.nanpercentile(y[np.where((x <= x_new[i + 1]) & (x >= x_new[i]))], 50)
    line, = ax.plot(x_new, y1, '-', color=color, linewidth=width, label=legend, linestyle=linestyle)

    return(ax,line)

if __name__ == '__main__':
    log_string = 'Log draw_ndvi.py : '

    # vegetation masks
    irrigation = read_data(data_path('study2/original_data/irrigation/gmia_v5_aei_pct_360_720.npy'))
    fvc = read_data(data_path('Proj1VD/original_data/Landcover/VCF5KYR/vcf5kyr_v001/VCF_1982_to_2016_0Tree_1nonTree_yearly.npy'))
    vegetation_cover = np.nanmean(fvc[0, :, :, :] + fvc[1, :, :, :], axis=0)

    # overall sensitivity of obs
    sensitivity_ens = read_data(data_path('study2/results/result_july/Ensemble-8obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_singleSMOUT.npy'))
    sensitivity_ens[:,3, :, :, :][sensitivity_ens[:,4,:,:,:]>0.05]=np.nan
    overall_sen1 = sensitivity_ens[7, 3, 1:3, :, :] # obs

    # overall sensitivity of model
    sensitivity_ens = read_data(data_path('study2/results/result_april/TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_SM13modelOUT.npy'))
    sensitivity_ens[:, 3, :, :, :][sensitivity_ens[:, 4, :, :, :] > 0.05] = np.nan
    overall_sen2 = sensitivity_ens[9, 3, 1:3, :, :]  # model
    overall_sen1[np.isnan(overall_sen2)] = np.nan
    overall_sen2[np.isnan(overall_sen1)] = np.nan

    # obs SM mean
    overall_SM1 = np.zeros((2,360,720))
    overall_SM1[0,:,:] = np.nanmean(read_data(data_path('study2/original_data/ERA5-Land/0d50_monthly/ESM-4obs_SMsurf_1982_2017_monthly_growseason_yearlymean.npy'))[:,3,:,:], axis=0)
    overall_SM1[1,:,:] = np.nanmean(read_data(data_path('study2/original_data/ERA5-Land/0d50_monthly/ESM-4obs_SMroot_1982_2017_monthly_growseason_yearlymean.npy'))[:,3,:,:], axis=0)
    # model SM mean
    SMsurf_ens = np.nanmean(read_data(data_path('study2/original_data/TRENDY_climate/v7_msl/TRENDYS3-10model_SMsurf_1982_2017_monthly_growseason_yearlymean.npy')),axis=0)
    SMroot_ens = np.nanmean(read_data(data_path('study2/original_data/TRENDY_climate/v7_msl/TRENDYS3-10model_SMroot_1982_2017_monthly_growseason_yearlymean.npy')),axis=0)
    overall_SM2 = np.zeros((2,360,720))
    overall_SM2[0,:,:] = SMsurf_ens[9,:,:]
    overall_SM2[1,:,:] = SMroot_ens[9,:,:]

    fig = plt.figure(figsize=[8, 6], dpi=300, tight_layout=True)
    gs = gridspec.GridSpec(3, 2, wspace=0.35, hspace=0.4)
    label = ['obs', 'model']
    color = ['blue', 'darkorange']
    # nn = 50
    nn = 30
    x_new = [np.linspace(0, nn, 40), np.linspace(0, 300, 40)]
    x_tick = [[0, nn/2, nn], [0, 150, 300]]
    x_lim = [[0, nn], [0, 300]]
    y_tick = [[-0.02,-0.01,0,0.01,0.02], [-0.01,0,0.01,0.02]]
    y_lim = [[-0.02, 0.02], [-0.01, 0.02]]

    for v in range(2):
        ax0 = fig.add_subplot(gs[v])
        if v==0:
            ax0.set_title('(A)')
        else:
            ax0.set_title('(D)')
        overall_sen1[v, :, :][overall_sen1[v, :, :] < np.round(np.nanpercentile(overall_sen1[v, :, :], 2), 3)] = np.round(np.nanpercentile(overall_sen1[v, :, :], 2), 3)
        overall_sen1[v, :, :][overall_sen1[v, :, :] > np.round(np.nanpercentile(overall_sen1[v, :, :], 98), 3)] = np.round(np.nanpercentile(overall_sen1[v, :, :], 98), 3)
        overall_sen1[v, :, :][vegetation_cover < 5] = np.nan
        overall_sen1[v, :, :][irrigation > 10] = np.nan
        overall_SM1[v, :, :][overall_SM1[v, :, :] < np.round(np.nanpercentile(overall_SM1[v, :, :], 2), 3)] = np.round(np.nanpercentile(overall_SM1[v, :, :], 2), 3)
        overall_SM1[v, :, :][overall_SM1[v, :, :] > np.round(np.nanpercentile(overall_SM1[v, :, :], 98), 3)] = np.round(np.nanpercentile(overall_SM1[v, :, :], 98), 3)
        overall_sen2[v, :, :][overall_sen2[v, :, :] < np.round(np.nanpercentile(overall_sen2[v, :, :], 2), 3)] = np.round(np.nanpercentile(overall_sen2[v, :, :], 2), 3)
        overall_sen2[v, :, :][overall_sen2[v, :, :] > np.round(np.nanpercentile(overall_sen2[v, :, :], 98), 3)] = np.round(np.nanpercentile(overall_sen2[v, :, :], 98), 3)
        overall_sen2[v, :, :][vegetation_cover < 5] = np.nan
        overall_sen2[v, :, :][irrigation > 10] = np.nan
        overall_SM2[v, :, :][overall_SM2[v, :, :] < np.round(np.nanpercentile(overall_SM2[v, :, :], 2), 3)] = np.round(np.nanpercentile(overall_SM2[v, :, :], 2), 3)
        overall_SM2[v, :, :][overall_SM2[v, :, :] > np.round(np.nanpercentile(overall_SM2[v, :, :], 98), 3)] = np.round(np.nanpercentile(overall_SM2[v, :, :], 98), 3)

        SM_new1 = np.concatenate(overall_SM1[v, :, :])
        Sensi_all_new1 = np.concatenate(overall_sen1[v, :, :])
        SM_new2 = np.concatenate(overall_SM2[v, :, :])
        Sensi_all_new2 = np.concatenate(overall_sen2[v, :, :])

        y_data1 = Sensi_all_new1[~np.isnan(Sensi_all_new1) & ~np.isnan(SM_new1)]
        x_data1 = SM_new1[~np.isnan(Sensi_all_new1) & ~np.isnan(SM_new1)]
        y_data2 = Sensi_all_new2[~np.isnan(Sensi_all_new2) & ~np.isnan(SM_new2)]
        x_data2 = SM_new2[~np.isnan(Sensi_all_new2) & ~np.isnan(SM_new2)]
        sm_max = max(np.nanmax(x_data1),np.nanmax(x_data2))
        print(sm_max)

        ax, ax1 =interquantile_plot(ax0, 40, x_new[v], x_data1, y_data1, 'obs', 'blue')
        ax, ax1 =interquantile_plot(ax0, 40, x_new[v], x_data2, y_data2, 'Model', 'darkorange')
        ax.axvline(x=0, color='r', linewidth=0.2)
        ax.axhline(y=0, color='r', linewidth=0.2)
        ax.set_ylabel(tit[v] + unit)
        ax.set_ylim(y_lim[v])
        ax.set_yticks(y_tick[v])
        ax.set_xlim(x_lim[v])
        ax.set_xticks(x_tick[v])
        ax1.set_ylabel('Probability')

    ax.legend(loc=(0.7,0.7), frameon=False) # loc: a pair of float, indicating percentage of position in a figure

    #############plot 7 obs
    ax20 = fig.add_subplot(gs[2])
    ax20.set_ylabel(tit[0] + unit)
    ax20.set_title('(B) obs')
    ax30 = fig.add_subplot(gs[3])
    ax30.set_ylabel(tit[1] + unit)
    ax30.set_title('(E) obs')
    color = ['#cc9900','#cccc00','#ff6600','#cc3300','#009900','#0066ff','#006600','#000000']
    width = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,1]

    # obs SM mean
    legend_list = ['LAI3g_EnSM','LAIltdr_EnSM','LAIglass_EnSM','LAIglobmap_EnSM','EnLAI_ERA5-land','EnLAI_Gleam','EnLAI_MERRA-2','EnLAI_EnSM']

    sensitivity_ens = read_data(data_path('study2/results/result_july/Ensemble-8obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_singleSMOUT.npy'))
    sensitivity_ens[:,3, :, :, :][sensitivity_ens[:,4, :, :, :] > 0.05] = np.nan

    sm_ind = [3,3,3,3,0,1,2,3]
    # for v in [0,1,2,3,4,5,6,7]:
    for v in [0, 1, 2, 3, 7]:
        linestyle = '-'
        # overall sensitivity of obs
        sen_surf = sensitivity_ens[v,3, 1, :, :]
        sen_root = sensitivity_ens[v,3, 2, :, :]

        # obs SM mean
        SM_surf = np.nanmean(read_data(data_path('study2/original_data/ERA5-Land/0d50_monthly/ESM-4obs_SMsurf_1982_2017_monthly_growseason_yearlymean.npy'))[:, sm_ind[v], :, :], axis=0)
        SM_root = np.nanmean(read_data(data_path('study2/original_data/ERA5-Land/0d50_monthly/ESM-4obs_SMroot_1982_2017_monthly_growseason_yearlymean.npy'))[:, sm_ind[v], :, :], axis=0)

        sen_surf[irrigation > 10] = np.nan
        sen_surf[vegetation_cover < 0.05] = np.nan
        sen_surf[sen_surf < np.round(np.nanpercentile(sen_surf, 2), 3)] = np.round(np.nanpercentile(sen_surf, 2), 3)
        sen_surf[sen_surf > np.round(np.nanpercentile(sen_surf, 98), 3)] = np.round(np.nanpercentile(sen_surf, 98), 3)
        sen_surf[np.isnan(overall_sen1[0,:,:])] = np.nan
        sen_root[irrigation > 10] = np.nan
        sen_root[vegetation_cover < 0.05] = np.nan
        sen_root[sen_root < np.round(np.nanpercentile(sen_root, 2), 3)] = np.round(np.nanpercentile(sen_root, 2), 3)
        sen_root[sen_root > np.round(np.nanpercentile(sen_root, 98),3)] = np.round(np.nanpercentile(sen_root, 98),3)
        sen_root[np.isnan(overall_sen1[1,:,:])] = np.nan

        SM_surf[SM_surf < np.round(np.nanpercentile(SM_surf, 2), 3)] = np.round(np.nanpercentile(SM_surf, 2), 3)
        SM_surf[SM_surf > np.round(np.nanpercentile(SM_surf, 98), 3)] = np.round(np.nanpercentile(SM_surf, 98), 3)
        SM_root[SM_root < np.round(np.nanpercentile(SM_root, 2), 3)] = np.round(np.nanpercentile(SM_root, 2), 3)
        SM_root[SM_root > np.round(np.nanpercentile(SM_root, 98), 3)] = np.round(np.nanpercentile(SM_root, 98), 3)

        y_data1 = sen_surf[~np.isnan(sen_surf) & ~np.isnan(SM_surf)]
        x_data1 = SM_surf[~np.isnan(sen_surf) & ~np.isnan(SM_surf)]
        y_data2 = sen_root[~np.isnan(sen_root) & ~np.isnan(SM_root)]
        x_data2 = SM_root[~np.isnan(sen_root) & ~np.isnan(SM_root)]

        if v==4 or v==5 or v==6:
            linestyle = '--'
            print(np.nanmax(x_data1), np.nanmax(x_data2), np.nanmax(y_data1), np.nanmax(y_data2))

        ax,line = interquantile_plot1(ax20, 40, x_new[0], x_data1, y_data1, color[v], width[v], '', linestyle)
        ax.set_ylim(y_lim[0])
        ax.set_yticks(y_tick[0])
        ax.set_xlim(x_lim[0])
        ax.set_xticks(x_tick[0])
        ax.set_ylabel(tit[0] + unit)
        ax.axvline(x=0, color='r', linewidth=0.2)
        ax.axhline(y=0, color='r', linewidth=0.2)

        # ax,line = interquantile_plot1(ax30, 40, np.linspace(0, 900, 40), x_data2, y_data2, color[v], width[v], legend_list[v], linestyle)
        ax,line = interquantile_plot1(ax30, 40, x_new[1], x_data2, y_data2, color[v], width[v], legend_list[v], linestyle)
        ax.set_ylim(-0.001, 0.003)
        ax.set_yticks([-0.001, 0, 0.001, 0.002, 0.003])
        # ax.set_xlim(0,900)
        # ax.set_xticks([0,450,900])
        ax.set_xlim(x_lim[1])
        ax.set_xticks(x_tick[1])
        ax.set_ylabel(tit[1] + unit)
        ax.axvline(x=0, color='r', linewidth=0.2)
        ax.axhline(y=0, color='r', linewidth=0.2)

    # ax.legend(loc=(0.6,0.4), frameon=False, fontsize=4) # loc: a pair of float, indicating percentage of position in a figure

    ############# plot 9 models #################
    ax40 = fig.add_subplot(gs[4])
    ax40.set_title('(C) Model')
    ax50 = fig.add_subplot(gs[5])
    ax50.set_title('(F) Model')
    color = ['#cc9900','#cccc00','#006600','#ff6600','#cc0066','#009900','#990099','#cc3300','#0066ff','#000000']
    width = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1]
    line_type = ['-','--']

    SMsurf_ens = np.nanmean(read_data(data_path('study2/original_data/TRENDY_climate/v7_msl/TRENDYS3-10model_SMsurf_1982_2017_monthly_growseason_yearlymean.npy')),axis=0)
    SMroot_ens = np.nanmean(read_data(data_path('study2/original_data/TRENDY_climate/v7_msl/TRENDYS3-10model_SMroot_1982_2017_monthly_growseason_yearlymean.npy')),axis=0)

    sensitivity_ens = read_data(data_path('study2/results/result_april/TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_SM13modelOUT.npy'))
    sensitivity_ens[:, 3, :, :, :][sensitivity_ens[:, 4, :, :, :] > 0.05] = np.nan
    overall_sen_ens_surf = sensitivity_ens[:, 3, 1, :, :]
    overall_sen_ens_root = sensitivity_ens[:, 3, 2, :, :]

    # for v in [0,1,2,3,4,5,6,7,8,9]:
    for v in [0, 2, 5, 6, 7,8,9]:
        linestyle = '-'
        sen_surf = overall_sen_ens_surf[v, :, :]
        SM_surf = SMsurf_ens[v, :, :]
        sen_root = overall_sen_ens_root[v, :, :]
        SM_root = SMroot_ens[v, :, :]
        sen_surf[irrigation > 10] = np.nan
        sen_surf[vegetation_cover < 5] = np.nan
        sen_surf[sen_surf < np.round(np.nanpercentile(sen_surf, 2), 3)] = np.round(np.nanpercentile(sen_surf, 2), 3)
        sen_surf[sen_surf > np.round(np.nanpercentile(sen_surf, 97), 3)] = np.round(np.nanpercentile(sen_surf, 97), 3)
        sen_surf[np.isnan(overall_sen1[0,:,:])] = np.nan
        sen_root[irrigation > 10] = np.nan
        sen_root[vegetation_cover < 5] = np.nan
        sen_root[sen_root < np.round(np.nanpercentile(sen_root, 2), 3)] = np.round(np.nanpercentile(sen_root, 2), 3)
        sen_root[sen_root > np.round(np.nanpercentile(sen_root, 98),3)] = np.round(np.nanpercentile(sen_root, 98),3)
        sen_root[np.isnan(overall_sen1[1,:,:])] = np.nan

        SM_surf[SM_surf < np.round(np.nanpercentile(SM_surf, 2), 3)] = np.round(np.nanpercentile(SM_surf, 2), 3)
        SM_surf[SM_surf > np.round(np.nanpercentile(SM_surf, 98), 3)] = np.round(np.nanpercentile(SM_surf, 98), 3)
        SM_root[SM_root < np.round(np.nanpercentile(SM_root, 2), 3)] = np.round(np.nanpercentile(SM_root, 2), 3)
        SM_root[SM_root > np.round(np.nanpercentile(SM_root, 98), 3)] = np.round(np.nanpercentile(SM_root, 98), 3)

        y_data1 = sen_surf[~np.isnan(sen_surf) & ~np.isnan(SM_surf)]
        x_data1 = SM_surf[~np.isnan(sen_surf) & ~np.isnan(SM_surf)]
        y_data2 = sen_root[~np.isnan(sen_root) & ~np.isnan(SM_root)]
        x_data2 = SM_root[~np.isnan(sen_root) & ~np.isnan(SM_root)]

        if v==1 or v==3 or v==4:
            linestyle = '--'

        # ax,line = interquantile_plot1(ax40, 40, np.linspace(0, 200, 40), x_data1, y_data1, color[v], width[v], model[v], linestyle)
        ax,line = interquantile_plot1(ax40, 40, x_new[0], x_data1, y_data1, color[v], width[v], model[v], linestyle)
        ax.set_ylim([-0.1,0.1])
        ax.set_yticks([-0.1,-0.05, 0, 0.05, 0.1])
        # ax.set_xlim(0,200)
        # ax.set_xticks([0,100,200])
        ax.set_xlim(x_lim[0])
        ax.set_xticks(x_tick[0])
        ax.set_xlabel(x_label[0])
        ax.set_ylabel(tit[0] + unit)
        ax.axvline(x=0, color='r', linewidth=0.2)
        ax.axhline(y=0, color='r', linewidth=0.2)
        # ax.legend(loc=(0.7, 0), frameon=False, fontsize=4)  # loc: a pair of float, indicating percentage of position in a figure

        ax,line = interquantile_plot1(ax50, 40, x_new[1], x_data2, y_data2, color[v], width[v], '', '-')
        ax.set_ylim(-0.02, 0.04)
        ax.set_yticks([-0.02, 0, 0.02, 0.04])
        ax.set_xlim(x_lim[1])
        ax.set_xticks(x_tick[1])
        ax.set_xlabel(x_label[1])
        ax.set_ylabel(tit[1] + unit)
        ax.axvline(x=0, color='r', linewidth=0.2)
        ax.axhline(y=0, color='r', linewidth=0.2)

    plt.savefig('/Net/Groups/BGI/scratch/wantong/study2/results/result_april/figure2/fig2.jpg', bbox_inches='tight')

    print('end')



