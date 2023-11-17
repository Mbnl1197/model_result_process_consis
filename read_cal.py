import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib as mpl
import os
import re
import pandas as pd
import netCDF4 as nc
import xarray as xr


# sm.bias(pred,ref)



def obs_value(var):
    obs = data_obs[var].values.flatten()
    obs = pd.Series(obs,index=time)
    obs = obs.replace(-9999,np.nan)
    # obs = obs.resample('D').mean()
    return obs
def mod_value(var,mdata):
    model = mdata[var].values.flatten()
    model = pd.Series(model,index=time)
    # model = model.resample('D').mean()
    return model

def remov_nan(obs,igbp,pft,pc):
    a = pd.concat([obs,igbp,pft,pc],axis=1)
    a.dropna(inplace=True)
    return a

def round_up(value):
    return np.round(value,1)

def rmse(pre,obs):
    rmse = np.sqrt(np.mean((np.array(pre) - np.array(obs)) ** 2))
    if not np.isnan(rmse):
        rmse = round_up(rmse)
    return rmse

def bias(pre,obs):
    b = np.mean(pre) - np.mean(obs)
    return b

def bias_rmse_corr(lh,h,r,gpp,ustar,outfile_name):
    lh_igbp_bias = bias(lh[1],lh[0])
    lh_pft_bias = bias(lh[2],lh[0])
    lh_pc_bias = bias(lh[3],lh[0])
    h_igbp_bias = bias(h[1],h[0])
    h_pft_bias = bias(h[2],h[0])
    h_pc_bias = bias(h[3],h[0])
    r_igbp_bias = bias(r[1],r[0])
    r_pft_bias = bias(r[2],r[0])
    r_pc_bias = bias(r[3],r[0])
    gpp_igbp_bias = bias(gpp[1],gpp[0])
    gpp_pft_bias = bias(gpp[2],gpp[0])
    gpp_pc_bias = bias(gpp[3],gpp[0])
    ustar_igbp_bias = bias(ustar[1],ustar[0])
    ustar_pft_bias = bias(ustar[2],ustar[0])
    ustar_pc_bias = bias(ustar[3],ustar[0])


    lh_igbp_rmse = rmse(lh[1],lh[0])
    lh_pft_rmse = rmse(lh[2],lh[0])
    lh_pc_rmse = rmse(lh[3],lh[0])
    h_igbp_rmse = rmse(h[1],h[0])
    h_pft_rmse = rmse(h[2],h[0])
    h_pc_rmse = rmse(h[3],h[0])
    r_igbp_rmse = rmse(r[1],r[0])
    r_pft_rmse = rmse(r[2],r[0])
    r_pc_rmse = rmse(r[3],r[0])
    gpp_igbp_rmse = rmse(gpp[1],gpp[0])
    gpp_pft_rmse = rmse(gpp[2],gpp[0])
    gpp_pc_rmse = rmse(gpp[3],gpp[0])
    ustar_igbp_rmse = rmse(ustar[1],ustar[0])
    ustar_pft_rmse = rmse(ustar[2],ustar[0])
    ustar_pc_rmse = rmse(ustar[3],ustar[0])


    lh_igbp_corr = np.corrcoef(lh[1], lh[0])[0,1]
    lh_pft_corr = np.corrcoef(lh[2], lh[0])[0,1]
    lh_pc_corr = np.corrcoef(lh[3], lh[0])[0,1]
    h_igbp_corr = np.corrcoef(h[1], h[0])[0,1]
    h_pft_corr = np.corrcoef(h[2], h[0])[0,1]
    h_pc_corr = np.corrcoef(h[3], h[0])[0,1]
    r_igbp_corr = np.corrcoef(r[1], r[0])[0,1]
    r_pft_corr = np.corrcoef(r[2], r[0])[0,1]
    r_pc_corr = np.corrcoef(r[3], r[0])[0,1]
    gpp_igbp_corr = np.corrcoef(gpp[1], gpp[0])[0,1]
    gpp_pft_corr = np.corrcoef(gpp[2], gpp[0])[0,1]
    gpp_pc_corr = np.corrcoef(gpp[3], gpp[0])[0,1]
    ustar_igbp_corr = np.corrcoef(ustar[1], ustar[0])[0,1]
    ustar_pft_corr = np.corrcoef(ustar[2], ustar[0])[0,1]
    ustar_pc_corr = np.corrcoef(ustar[3], ustar[0])[0,1]



    bias_rmse_corr_time = pd.DataFrame(columns=['lh_igbp_bias','lh_pft_bias','lh_pc_bias',
                                'h_igbp_bias','h_pft_bias','h_pc_bias',
                                'r_igbp_bias','r_pft_bias','r_pc_bias',
                                'gpp_igbp_bias','gpp_pft_bias','gpp_pc_bias',
                                'ustar_igbp_bias','ustar_pft_bias','ustar_pc_bias',

                                'lh_igbp_rmse','lh_pft_rmse','lh_pc_rmse',
                                'h_igbp_rmse','h_pft_rmse','h_pc_rmse',
                                'r_igbp_rmse','r_pft_rmse','r_pc_rmse',
                                'gpp_igbp_rmse','gpp_pft_rmse','gpp_pc_rmse',
                                'ustar_igbp_rmse','ustar_pft_rmse','ustar_pc_rmse',
                                
                                'lh_igbp_corr','lh_pft_corr','lh_pc_corr',
                                'h_igbp_corr','h_pft_corr','h_pc_corr',
                                'r_igbp_corr','r_pft_corr','r_pc_corr',
                                'gpp_igbp_corr','gpp_pft_corr','gpp_pc_corr',
                                'ustar_igbp_corr','ustar_pft_corr','ustar_pc_corr',
    ])

    bias_rmse_corr_time.loc['AU-How'] = [lh_igbp_bias,lh_pft_bias,lh_pc_bias,
                                    h_igbp_bias,h_pft_bias,h_pc_bias,
                                    r_igbp_bias,r_pft_bias,r_pc_bias,
                                    gpp_igbp_bias,gpp_pft_bias,gpp_pc_bias,
                                    ustar_igbp_bias,ustar_pft_bias,ustar_pc_bias,
                                    
                                    lh_igbp_rmse,lh_pft_rmse,lh_pc_rmse,
                                    h_igbp_rmse,h_pft_rmse,h_pc_rmse,
                                    r_igbp_rmse,r_pft_rmse,r_pc_rmse,
                                    gpp_igbp_rmse,gpp_pft_rmse,gpp_pc_rmse,
                                    ustar_igbp_rmse,ustar_pft_rmse,ustar_pc_rmse,
                                    
                                    lh_igbp_corr,lh_pft_corr,lh_pc_corr,
                                    h_igbp_corr,h_pft_corr,h_pc_corr,
                                    r_igbp_corr,r_pft_corr,r_pc_corr,
                                    gpp_igbp_corr,gpp_pft_corr,gpp_pc_corr,
                                    ustar_igbp_corr,ustar_pft_corr,ustar_pc_corr]
    bias_rmse_corr_time.to_csv(f'/stu01/shijh21/liuz/sis/{outfile_name}')





obsdir = '/stu01/shijh21/data/forcingPLUMBER2/flux/'
obsfile = obsdir + 'AU-How_2003-2017_OzFlux_Flux.nc'
moddir = '/stu01/shijh21/liuz/sis/'
igbp_file = moddir + 'AU-How_IGBP/history/AU-How_IGBP_hist_2017.nc'
pft_file = moddir + 'AU-How_PFT/history/AU-How_PFT_hist_2017.nc'
pc_file = moddir + 'AU-How_PC/history/AU-How_PC_hist_2017.nc'
data_obs = xr.open_dataset(obsfile)
time = data_obs.time

lh_obs = obs_value('Qle_cor')
h_obs = obs_value('Qh_cor')
r_obs = obs_value('Rnet')
gpp_obs = obs_value('GPP') #"umol/m2/s"
gpp_obs = gpp_obs*12*86400*0.000001
ustar_obs = obs_value('Ustar')

igbp = xr.open_dataset(igbp_file)
pft = xr.open_dataset(pft_file)
pc = xr.open_dataset(pc_file)
time = pd.date_range('2017-01-01T00:00:00','2017-12-31T23:30:00',freq='0.5H')

lh_igbp = mod_value('f_lfevpa',igbp)
h_igbp = mod_value('f_fsena',igbp)
r_igbp = mod_value('f_rnet',igbp)
gpp_igbp = mod_value('f_assim',igbp) # "mol m-2 s-1"
gpp_igbp = gpp_igbp*12*86400
ustar_igbp = mod_value('f_ustar',igbp)

lh_pft = mod_value('f_lfevpa',pft)
h_pft = mod_value('f_fsena',pft)
r_pft = mod_value('f_rnet',pft)
gpp_pft = mod_value('f_assim',pft) # "mol m-2 s-1"
gpp_pft = gpp_pft*12*86400
ustar_pft = mod_value('f_ustar',pft)

lh_pc = mod_value('f_lfevpa',pc)
h_pc = mod_value('f_fsena',pc)
r_pc = mod_value('f_rnet',pc)
gpp_pc = mod_value('f_assim',pc) # "mol m-2 s-1"
gpp_pc = gpp_pc*12*86400
ustar_pc = mod_value('f_ustar',pc)


lh = remov_nan(lh_obs,lh_igbp,lh_pft,lh_pc)
h  = remov_nan(h_obs,h_igbp,h_pft,h_pc)
r = remov_nan(r_obs,r_igbp,r_pft,r_pc)
gpp = remov_nan(gpp_obs,gpp_igbp,gpp_pft,gpp_pc)
ustar = remov_nan(ustar_obs,ustar_igbp,ustar_pft,ustar_pc)

##################30min#################
# def bias_rmse_corr(lh,h,r,gpp,ustar,outfile_name):
bias_rmse_corr(lh,h,r,gpp,ustar,'bias_rmse_corr_30min.csv')

#######################30min end################################

####################daily######################################

lh = lh.resample('D').mean()
h = h.resample('D').mean()
r = r.resample('D').mean()
gpp = gpp.resample('D').mean()
ustar = ustar.resample('D').mean()
bias_rmse_corr(lh,h,r,gpp,ustar,'bias_rmse_corr_day.csv')
#######################  daily  end  ###################################3

##################以日平均为基础，在每月求bias，rmse，r###################################
for month in [1,2,3,4,5,6,7,8,9,10,11,12]:

    # lh[f'2017-{month}']
    # h[f'2017-{month}']
    # r[f'2017-{month}']
    # gpp[f'2017-{month}']
    # ustar[f'2017-{month}']
    bias_rmse_corr(lh[f'2017-{month}'],h[f'2017-{month}'],r[f'2017-{month}'],
                   gpp[f'2017-{month}'],ustar[f'2017-{month}'],
                   f'bias_rmse_corr_{month}month.csv')

###############################  每月bias，rmse，r；  end   ###################################


























