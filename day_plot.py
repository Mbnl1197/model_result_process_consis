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


lh = lh.resample('D').mean()
h = h.resample('D').mean()
r = r.resample('D').mean()
gpp = gpp.resample('D').mean()
ustar = ustar.resample('D').mean()


fig,axes = plt.subplots(5,figsize = (13,13),
                        sharex=True,sharey='row')


x = np.arange(365)

line_obs = axes[0].plot(x,lh[0],c = 'gray',label = 'OBS',linewidth = 1)
line_igbp = axes[0].plot(x,lh[1], c = 'red',label = 'IGBP',linewidth = 1)
line_pft = axes[0].plot(x,lh[2],c = 'green',label = 'PFT',linewidth = 1)
line_pc = axes[0].plot(x,lh[3],c = 'blue',label = 'PC',linewidth = 1)


axes[1].plot(x,h[0],c = 'gray',linewidth = 1)
axes[1].plot(x,h[1], c = 'red',linewidth = 1)
axes[1].plot(x,h[2],c = 'green',linewidth = 1)
axes[1].plot(x,h[3],c = 'blue',linewidth = 1)

axes[2].plot(x,r[0],c = 'gray',linewidth = 1)
axes[2].plot(x,r[1], c = 'red',linewidth = 1)
axes[2].plot(x,r[2],c = 'green',linewidth = 1)
axes[2].plot(x,r[3],c = 'blue',linewidth = 1)

axes[3].plot(x,gpp[0],c = 'gray',linewidth = 1)
axes[3].plot(x,gpp[1], c = 'red',linewidth = 1)
axes[3].plot(x,gpp[2],c = 'green',linewidth = 1)
axes[3].plot(x,gpp[3],c = 'blue',linewidth = 1)

axes[4].plot(x,ustar[0],c = 'gray',linewidth = 1)
axes[4].plot(x,ustar[1], c = 'red',linewidth = 1)
axes[4].plot(x,ustar[2],c = 'green',linewidth = 1)
axes[4].plot(x,ustar[3],c = 'blue',linewidth = 1)


ticks = axes[4].set_xticks([0,31,59,90,120,151,181,212,243,273,304,334])
labels = axes[4].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jan',
                                    'Jul','Aug','Sep','Oct','Nov','Dec'],
                                    rotation = 30,fontsize = 10)
axes[0].set_ylabel('LH',fontsize = 20)
axes[1].set_ylabel('H',fontsize = 20)
axes[2].set_ylabel('Rnet',fontsize = 20)
axes[3].set_ylabel('GPP',fontsize = 20)
axes[4].set_ylabel('Ustar',fontsize = 20)




fig.legend(bbox_to_anchor=(0.7, 0.96),prop = {'size':15})
fig.suptitle(f'AU-How',fontsize = 20)
plt.tight_layout()
plt.show()






























