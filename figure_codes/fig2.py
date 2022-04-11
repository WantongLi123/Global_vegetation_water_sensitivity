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

tit = [r'$\frac{\partial{LAI}}{\partial{SMnear}}$',r'$\frac{\partial{LAI}}{\partial{SMsub}}$']
unit = ' ($\mathregular{mm^{-1}}$)'
x_label = ['Growing-season SMnear (mm)','Growing-season SMsub (mm)']
model = ['ISAM','LPJ-GUESS','LPX-Bern','VISIT','CABLE-POP','CLM5.0','JSBACH','JULES','ORCHIDEE-CNP','Ensemble model mean']

def data_path(filename):
    file_path = "{path}/{filename}".format(
        path="...your_path...",
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
    ax1.set_ylim([0, 10000])
    ax1.get_yaxis().set_ticks([])
    return(ax,ax1)

def interquantile_plot1(ax, n, x_new, x, y, color, width, legend, linestyle):
    y1 = np.zeros(n) * np.nan
    for i in range(n - 1):
        y1[i] = np.nanpercentile(y[np.where((x <= x_new[i + 1]) & (x >= x_new[i]))], 50)
    line, = ax.plot(x_new, y1, '-', color=color, linewidth=width, label=legend, linestyle=linestyle)

    return(ax,line)

if __name__ == '__main__':
    log_string = 'Log: '

    # vegetation masks
    irrigation = read_data(data_path('gmia_v5_aei_pct_360_720.npy'))
    fvc = read_data(data_path('VCF_1982_to_2016_0Tree_1nonTree_yearly.npy'))
    vegetation_cover = np.nanmean(fvc[0, :, :, :] + fvc[1, :, :, :], axis=0)

    # overall sensitivity of obs
    sensitivity_ens = read_data(data_path('Ensemble-6obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear.npy'))
    sensitivity_ens[:,3, :, :, :][sensitivity_ens[:,4,:,:,:]>0.01]=np.nan
    # sensitivity_ens[:,3, :, :, :][sensitivity_ens[:,3,:,:,:]<0]=np.nan
    overall_sen1 = sensitivity_ens[5, 3, 1:3, :, :] # obs

    # overall sensitivity of model
    sensitivity_ens = read_data(data_path('TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_SM13modelOUT.npy'))
    sensitivity_ens[:, 3, :, :, :][sensitivity_ens[:, 4, :, :, :] > 0.01] = np.nan
    # sensitivity_ens[:,3, :, :, :][sensitivity_ens[:,3,:,:,:]<0]=np.nan
    overall_sen2 = sensitivity_ens[9, 3, 1:3, :, :]  # model
    overall_sen1[np.isnan(overall_sen2)] = np.nan
    overall_sen2[np.isnan(overall_sen1)] = np.nan

    # obs SM mean
    overall_SM1 = np.zeros((2,360,720))
    # ERA5-land
    overall_SM1[0, :, :] = np.nanmean(read_data(data_path('ESM-mm-4obs_SMsurf_1982_2017_monthly_growseason_yearlymean.npy'))[:, 0, :, :], axis=0)
    overall_SM1[1, :, :] = np.nanmean(read_data(data_path('ESM-mm-4obs_SMroot_1982_2017_monthly_growseason_yearlymean.npy'))[:, 0, :, :], axis=0)
    # model SM mean
    SMsurf_ens = np.nanmean(read_data(data_path('TRENDYS3-10model_SMsurf_1982_2017_monthly_growseason_yearlymean.npy')),axis=0)
    SMroot_ens = np.nanmean(read_data(data_path('TRENDYS3-10model_SMroot_1982_2017_monthly_growseason_yearlymean.npy')),axis=0)
    overall_SM2 = np.zeros((2,360,720))
    overall_SM2[0,:,:] = SMsurf_ens[9,:,:]
    overall_SM2[1,:,:] = SMroot_ens[9,:,:]

    fig = plt.figure(figsize=[8, 2], dpi=300, tight_layout=True)
    gs = gridspec.GridSpec(1, 2, wspace=0.35, hspace=0.4)
    label = ['Obs', 'Model']
    color = ['black', 'darkorange']
    nn1 = 40
    nn2 = 600
    x_new = [np.linspace(0, nn1, 40), np.linspace(0, nn2, 40)]
    x_tick = [[0, nn1/2, nn1], [0, nn2/2, nn2]]
    x_lim = [[0, nn1], [0, nn2]]
    y_tick = [[-0.02,-0.01,0,0.01,0.02], [-0.01,0,0.01,0.02]]
    y_lim = [[-0.02, 0.02], [-0.01, 0.02]]

    for v in range(2):
        ax0 = fig.add_subplot(gs[v])
        if v==0:
            ax0.set_title('(a)')
        else:
            ax0.set_title('(b)')
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

        ax, ax1 =interquantile_plot(ax0, 40, x_new[v], x_data1, y_data1, 'Obs', 'black')
        ax, ax1 =interquantile_plot(ax0, 40, x_new[v], x_data2, y_data2, 'Model', 'darkorange')
        ax.axvline(x=0, color='r', linewidth=0.2)
        ax.axhline(y=0, color='r', linewidth=0.2)
        ax.set_ylabel(tit[v] + unit)
        ax.set_ylim(y_lim[v])
        ax.set_yticks(y_tick[v])
        ax.set_xlim(x_lim[v])
        ax.set_xticks(x_tick[v])
        ax.set_xlabel(x_label[v])
        ax1.set_ylabel('Probability')

    ax.legend(loc=(0.7,0.7), frameon=False) # loc: a pair of float, indicating percentage of position in a figure


    plt.savefig(data_path('fig2.jpg'), bbox_inches='tight')

    print('end')



