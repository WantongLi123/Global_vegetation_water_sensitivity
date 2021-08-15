#!/usr/bin/env python
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import sys
import scipy
from sklearn.linear_model import LinearRegression, TheilSenRegressor
import regressors.stats as regressors_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pymannkendall as mk
from sklearn.ensemble import RandomForestRegressor
import shap
from rfpimp import *
import seaborn as sns
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
plt.rcParams.update({'font.size': 4})
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['lines.markersize'] = 0.2

mutli_vari = ['Trends of mean LAI', 'Trends of maximum LAI', 'Trends of mean T', 'Trends of maximum T',
              'Trends of mean SMsurf', 'Trends of mean SMroot', 'Trends of mean VPD', 'Trends of mean SW', 'Trends of total P',
              'Short-to-tall veg ratio', 'Trends of short-to-tall veg ratio', 'Overall sensitivity to SMsurf','Overall sensitivity to SMroot',
                'Trends of sensitivity']


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

def yearlymean(lai, t2m, variable):
    yearly_mean = np.zeros((36, 360, 720)) * np.nan
    variable[lai<=0.5] = np.nan
    variable[t2m<=278.15] = np.nan

    for row in range(360):
        for col in range(720):
            if np.isnan(variable[:, row, col]).all():
                yearly_mean[:, row, col] = np.nan
            else:
                for y in range(36):
                    yearly_mean[y, row, col] = np.nanmean(variable[y*12:(y*12+12), row, col])
    return (yearly_mean)

def yearlymax(lai, t2m, variable):
    yearly_max = np.zeros((36, 360, 720)) * np.nan
    variable[lai<=0.5] = np.nan
    variable[t2m<=278.15] = np.nan

    for row in range(360):
        for col in range(720):
            if np.isnan(variable[:, row, col]).all():
                yearly_max[:, row, col] = np.nan
            else:
                for y in range(36):
                    yearly_max[y, row, col] = np.nanmax(variable[y*12:(y*12+12), row, col])
    return (yearly_max)

def yearlytotal(lai, t2m, variable):
    yearly_total = np.zeros((36, 360, 720)) * np.nan
    variable[lai<=0.5] = np.nan
    variable[t2m<=278.15] = np.nan

    for row in range(360):
        for col in range(720):
            if np.isnan(variable[:, row, col]).all():
                yearly_total[:, row, col] = np.nan
            else:
                for y in range(36):
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
                        slope[row, col] = result.slope
    return (slope)

def relative_imp(target, multi_predictor, xlim1, xlim2):
    X = multi_predictor.reshape(13, -1)
    X = np.transpose(X)
    y = target.reshape(-1)

    test_data = np.zeros((np.shape(y)[0],14)) * np.nan
    test_data[:,0:13] = X
    test_data[:,13] = y
    test_data = test_data[~np.all(test_data == 0, axis=1)]
    test_data = test_data[~np.any(np.isnan(test_data), axis=1)]
    test_data = test_data[~np.any(np.isinf(test_data), axis=1)]
    # for v in range(14):
    #     print(np.nanmin(test_data[:,v]),np.nanmax(test_data[:,v]))

    print('length of data:', len(test_data[:, 0]))

    output = np.nan
    if len(test_data[:, 0])>100:
        df_test = pd.DataFrame(
            {mutli_vari[0]: test_data[:, 0], mutli_vari[1]: test_data[:, 1],
             mutli_vari[2]: test_data[:, 2], mutli_vari[3]: test_data[:, 3],
             mutli_vari[4]: test_data[:, 4],
             mutli_vari[5]: test_data[:, 5], mutli_vari[6]: test_data[:, 6],
             mutli_vari[7]: test_data[:, 7],
             mutli_vari[8]: test_data[:, 8], mutli_vari[9]: test_data[:, 9],
             mutli_vari[10]: test_data[:, 10],
             mutli_vari[11]: test_data[:, 11],
             mutli_vari[12]: test_data[:, 12],
             mutli_vari[13]: test_data[:, 13]})
        X_test, y_test = df_test.drop('Trends of sensitivity', axis=1), df_test['Trends of sensitivity']
        print(X_test.head(100))

        rf = RandomForestRegressor(n_estimators=100,
                                       max_features=0.3,
                                       n_jobs=8,
                                       bootstrap=True,
                                       oob_score=True,
                                       random_state=42)

        rf.fit(X_test, y_test)
        print(rf.oob_score_)

        if rf.oob_score_ > 0:
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer(X_test)
            # shap_values = explainer(X_test.head(100))
            print(shap_values)
            shap.plots.beeswarm(shap_values, xlim1, xlim2)
            # output = np.nanmean(abs(shap_values), axis=0)  # Mean[abs(positive and negative SHAP)]---->importance
            # print(output)

            # print(importances(rf,X_test, y_test))
    return(ax)

if __name__ == '__main__':
    log_string = 'data-processing :'

    # LAI = read_data(data_path('study2/original_data/LAI/ELAI_4mean_1982_2018_monthly_0.5.npy'))
    #
    # TP = read_data('/Net/Groups/BGI/scratch/wantong/study2/original_data/TRENDY_climate/CRUJRApre_444_0.5.npy')  # mm
    # SSRD = read_data('/Net/Groups/BGI/scratch/wantong/study2/original_data/TRENDY_climate/CRUJRAdswrf_444_0.5.npy') / (1000000 / 86000)  # W/m2 to MJ/m2 day
    # T2M = read_data('/Net/Groups/BGI/scratch/wantong/study2/original_data/TRENDY_climate/CRUJRAtmp_444_0.5.npy')
    # VPD = read_data('/Net/Groups/BGI/scratch/wantong/study2/original_data/TRENDY_climate/CRUJRAvpd_444_0.5.npy')
    #
    # SM1 = read_data(data_path('study2/original_data/ERA5-Land/0d50_monthly/ESM-mm_SMsurf_1982_2017_monthly_0d50.npy'))
    # SM2 = read_data(data_path('study2/original_data/ERA5-Land/0d50_monthly/ESM-mm_SMroot_1982_2017_monthly_0d50.npy'))
    # SM3 = read_data(data_path('study2/original_data/TRENDY_climate/v7_msl/TRENDY_S3_SMsurf_1982_2017_monthly_SM_0d50.npy'))
    # SM4 = read_data(data_path('study2/original_data/TRENDY_climate/v7_msl/TRENDY_S3_SMroot_1982_2017_monthly_SM_0d50.npy'))
    # SM1[SM1 < 0] = np.nan
    # SM2[SM2 < 0] = np.nan
    # SM3[SM3 < 0] = np.nan
    # SM4[SM4 < 0] = np.nan
    #
    # LAI[LAI < 0] = np.nan  # NDVI have some values like -9999,-2499, needed to be removed
    # LAI[LAI > 20] = np.nan
    # TP[TP < 0] = np.nan
    # TP[TP > 10000] = np.nan
    # SM1[SM1 < 0] = np.nan
    # SM1[SM1 > 10000] = np.nan
    # SM2[SM2 < 0] = np.nan
    # SM2[SM2 > 10000] = np.nan
    # T2M[T2M < 0] = np.nan # Kelvin scale
    # T2M[T2M > 500] = np.nan
    # SSRD[SSRD < 0] = np.nan
    # SSRD[SSRD > 1000] = np.nan
    # VPD[VPD < 0] = np.nan
    # VPD[VPD > 50] = np.nan #VPD from ERA5 ranging 0-50; VPD from CRUJRA ranging 0-10
    #
    # fvc = read_data(data_path('Proj1VD/original_data/Landcover/VCF5KYR/vcf5kyr_v001/VCF_1982_to_2016_0Tree_1nonTree_yearly.npy'))
    # SM_trend = read_data(data_path('study2/results/result_july/ESM-obs-model_SM12_1982_2017_growseason_yearlymean_mkSlope.npy'))  # from index 0 to 3: SMsurf_obs, SMroot_obs, SMsurf_model, SMroot_model
    # over_sensi_obs = read_data(data_path('study2/results/result_july/Ensemble-8obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_singleSMOUT.npy'))[7, 3, 1:3, :, :]
    # over_p_obs = read_data(data_path('study2/results/result_july/Ensemble-8obs_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_singleSMOUT.npy'))[7, 4, 1:3, :, :]
    # over_sensi_model = read_data(data_path('study2/results/result_april/TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_SM13modelOUT.npy'))[9, 3, 1:3, :, :]
    # over_p_model = read_data(data_path('study2/results/result_april/TRENDYS3-10model_6v_monthly_1982_2017_0d50_ContributionSensitivity_allYear_SM13modelOUT.npy'))[9, 4, 1:3, :, :]
    # over_sensi_obs[over_sensi_obs <= 0] = np.nan
    # over_sensi_obs[over_p_obs >= 0.01] = np.nan
    # over_sensi_model[over_sensi_model <= 0] = np.nan
    # over_sensi_model[over_p_model >= 0.01] = np.nan
    # over_sensi_model[np.isnan(over_sensi_obs)] = np.nan
    # over_sensi_obs[np.isnan(over_sensi_model)] = np.nan
    #
    #
    # # data order: Trends of growing-season LAI mean, LAI max, T mean, T max, SMsurf, SMroot, VPD, SSRD, total P,
    # # short-vegetation ratio, trends of short-vegetation ratio, overall sensitivity to SMsurf, to SMroot
    # LAI_mean = yearlymean(LAI[0:432,:,:], T2M[0:432,:,:], LAI[0:432,:,:])
    # LAI_max= yearlymax(LAI[0:432,:,:], T2M[0:432,:,:], LAI[0:432,:,:])
    # T_mean = yearlymean(LAI[0:432, :, :], T2M[0:432, :, :], T2M[0:432, :, :])
    # T_max = yearlymax(LAI[0:432, :, :], T2M[0:432, :, :], T2M[0:432, :, :])
    # VPD_mean = yearlymean(LAI[0:432,:,:], T2M[0:432,:,:], VPD[0:432,:,:])
    # SSRD_mean = yearlymean(LAI[0:432,:,:], T2M[0:432,:,:], SSRD[0:432,:,:])
    # TP_mean = yearlytotal(LAI[0:432, :, :], T2M[0:432, :, :], TP[0:432, :, :])
    # grass_fraction_long_mean = np.nanmean(fvc[1,:,:,:], axis=0)/np.nanmean(fvc[0,:,:,:], axis=0)
    # grass_fraction_mean = fvc[1,:,:,:]/fvc[0,:,:,:]
    #
    # multiTrends_obs = np.zeros((13,360,720)) * np.nan
    # multiTrends_obs[0,:,:] = slope_yearly(LAI_mean)
    # multiTrends_obs[1,:,:] = slope_yearly(LAI_max)
    # multiTrends_obs[2,:,:] = slope_yearly(T_mean)
    # multiTrends_obs[3,:,:] = slope_yearly(T_max)
    # multiTrends_obs[4,:,:] = SM_trend[0,:,:]
    # multiTrends_obs[5,:,:] = SM_trend[1,:,:]
    # multiTrends_obs[6,:,:] = slope_yearly(VPD_mean)
    # multiTrends_obs[7,:,:] = slope_yearly(SSRD_mean)
    # multiTrends_obs[8,:,:] = slope_yearly(TP_mean)
    # multiTrends_obs[9,:,:] = grass_fraction_long_mean
    # multiTrends_obs[10,:,:] = slope_yearly(grass_fraction_mean)
    # multiTrends_obs[11,:,:] = over_sensi_obs[0,:,:]
    # multiTrends_obs[12,:,:] = over_sensi_obs[1,:,:]
    #
    # multiTrends_model = np.zeros((13, 360, 720)) * np.nan
    # multiTrends_model[0, :, :] = multiTrends_obs[0,:,:]
    # multiTrends_model[1, :, :] = multiTrends_obs[1,:,:]
    # multiTrends_model[2, :, :] = multiTrends_obs[2,:,:]
    # multiTrends_model[3, :, :] = multiTrends_obs[3,:,:]
    # multiTrends_model[4, :, :] = SM_trend[2, :, :]
    # multiTrends_model[5, :, :] = SM_trend[3, :, :]
    # multiTrends_model[6, :, :] = multiTrends_obs[6,:,:]
    # multiTrends_model[7, :, :] = multiTrends_obs[7,:,:]
    # multiTrends_model[8, :, :] = multiTrends_obs[8,:,:]
    # multiTrends_model[9, :, :] = multiTrends_obs[9,:,:]
    # multiTrends_model[10, :, :] = multiTrends_obs[10,:,:]
    # multiTrends_model[11, :, :] = over_sensi_model[0, :, :]
    # multiTrends_model[12, :, :] = over_sensi_model[1, :, :]
    # multiTrends_obs[9, :, :][multiTrends_obs[9, :, :] < 0] = np.nan
    # multiTrends_obs[9, :, :][multiTrends_obs[9, :, :] > 100] = np.nan
    # multiTrends_obs[10, :, :][multiTrends_obs[9, :, :] < 0] = np.nan
    # multiTrends_obs[10, :, :][multiTrends_obs[9, :, :] > 100] = np.nan
    # multiTrends_model[9, :, :][multiTrends_model[9, :, :] < 0] = np.nan
    # multiTrends_model[9, :, :][multiTrends_model[9, :, :] > 100] = np.nan
    # multiTrends_model[10, :, :][multiTrends_model[9, :, :] < 0] = np.nan
    # multiTrends_model[10, :, :][multiTrends_model[9, :, :] > 100] = np.nan
    # np.save(data_path('study2/results/result_july/multi_trends_13vari_sfig5_obs'), multiTrends_obs)
    # np.save(data_path('study2/results/result_july/multi_trends_13vari_sfig5_model'), multiTrends_model)

    multiTrends_obs = read_data(data_path('study2/results/result_july/multi_trends_13vari_sfig5_obs.npy'))
    multiTrends_model = read_data(data_path('study2/results/result_july/multi_trends_13vari_sfig5_model.npy'))
    Sensi_trend_obs = read_data('/Net/Groups/BGI/scratch/wantong/study2/results/result_july/ELAI_SM12_ERA5-land_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy')[0,:,:,:]
    Sensi_trend_obs_surf = Sensi_trend_obs[0,:,:]
    Sensi_trend_obs_root = Sensi_trend_obs[1,:,:]
    Sensi_trend_model = read_data('/Net/Groups/BGI/scratch/wantong/study2/results/result_july/TRENDY_S3_SM12_monthly_1982_2017_0d50_3Yblock_SensiTrend.npy')[0, :, :, :]
    Sensi_trend_model_surf = Sensi_trend_model[0, :, :]
    Sensi_trend_model_root = Sensi_trend_model[1, :, :]


    fig = plt.figure(figsize=[8, 8], dpi=300, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    xlim1, xlim2 = -0.001, 0.001 #obs surf
    # xlim1, xlim2 = -0.0001, 0.0001 #obs root
    # xlim1, xlim2 = -0.001, 0.001 #model surf
    # xlim1, xlim2 = -0.0004, 0.0004 #model root
    relative_imp(Sensi_trend_obs_surf, multiTrends_obs, xlim1, xlim2)
    plt.savefig(data_path('study2/results/result_april/figure2/figs5_surfobs.jpg'), bbox_inches='tight')


    print('end')