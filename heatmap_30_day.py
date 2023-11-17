

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib as mpl
import os
import re
import pandas as pd
import netCDF4 as nc
import xarray as xr
import matplotlib.colors as mcolors


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # ax.annotate('SDL', xy=(-0.2, 0.92), xytext=(-0.3, 0.92),
    #             #  xycoords   = 'axes fraction', 
    #              xycoords='axes fraction', 

    #             fontsize=15, ha='center', va='center',
    #             # bbox=dict(boxstyle='square', fc='white'),
    #             arrowprops=dict(arrowstyle='-[, widthB=5, lengthB=0.5', lw=2.0))


    # Set the desired range for the colorbar (replace min_val and max_val with your values)
    min_val = -50.0
    max_val = 50.0
    im.set_clim(vmin=min_val, vmax=max_val)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax,shrink=0.4, aspect=30, fraction=0.05,**cbar_kw)
    cbar = ax.figure.colorbar(im, ax=ax, aspect=30, fraction=0.05,**cbar_kw) 
    label = cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontsize = 30)
    label.set_size(20)
    cbar.ax.yaxis.set_label_coords(2.8, 0.5)


    cbar.set_ticks([-50,-40, -20, 0, 20, 40, 50])  # Set the ticks
    cbar.set_ticklabels(['-50','-40', '-20', '0', '20', '40','50'],fontsize = 20)  # Set the tick labels

    # cbar.ax.text(3.2, 54, 'ΔRMSE > 0', ha='center', va='center',size = 13)
    # cbar.ax.text(3.2, -54, 'ΔRMSE < 0', ha='center', va='center',size = 13)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels,size = 20)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels,size = 20)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-60, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar,ax



def heatmap_rmse(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # ax.annotate('SDL', xy=(-0.2, 0.92), xytext=(-0.3, 0.92),
    #             #  xycoords   = 'axes fraction', 
    #              xycoords='axes fraction', 

    #             fontsize=15, ha='center', va='center',
    #             # bbox=dict(boxstyle='square', fc='white'),
    #             arrowprops=dict(arrowstyle='-[, widthB=5, lengthB=0.5', lw=2.0))


    # Set the desired range for the colorbar (replace min_val and max_val with your values)
    min_val = 0
    max_val = 150.0
    im.set_clim(vmin=min_val, vmax=max_val)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax,shrink=0.4, aspect=30, fraction=0.05,**cbar_kw)
    cbar = ax.figure.colorbar(im, ax=ax, aspect=30, fraction=0.05,**cbar_kw) 
    label = cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontsize = 30)
    label.set_size(20)
    cbar.ax.yaxis.set_label_coords(2.8, 0.5)


    cbar.set_ticks([0,30, 60, 90, 120, 150,])  # Set the ticks
    cbar.set_ticklabels(['0','30', '60', '90', '120', '150'],fontsize = 20)  # Set the tick labels

    # cbar.ax.text(3.2, 54, 'ΔRMSE > 0', ha='center', va='center',size = 13)
    # cbar.ax.text(3.2, -54, 'ΔRMSE < 0', ha='center', va='center',size = 13)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels,size = 20)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels,size = 20)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-60, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar,ax

def heatmap_corr(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # ax.annotate('SDL', xy=(-0.2, 0.92), xytext=(-0.3, 0.92),
    #             #  xycoords   = 'axes fraction', 
    #              xycoords='axes fraction', 

    #             fontsize=15, ha='center', va='center',
    #             # bbox=dict(boxstyle='square', fc='white'),
    #             arrowprops=dict(arrowstyle='-[, widthB=5, lengthB=0.5', lw=2.0))


    # Set the desired range for the colorbar (replace min_val and max_val with your values)
    min_val = 0.5
    max_val = 1
    im.set_clim(vmin=min_val, vmax=max_val)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax,shrink=0.4, aspect=30, fraction=0.05,**cbar_kw)
    cbar = ax.figure.colorbar(im, ax=ax, aspect=30, fraction=0.05,**cbar_kw) 
    label = cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontsize = 30)
    label.set_size(20)
    cbar.ax.yaxis.set_label_coords(2.8, 0.5)


    cbar.set_ticks([0.5,0.6,0.7,0.8,0.9,1])  # Set the ticks
    cbar.set_ticklabels(['0.5','0.6', '0.7', '0.8','0.9', '1', ],fontsize = 20)  # Set the tick labels

    # cbar.ax.text(3.2, 54, 'ΔRMSE > 0', ha='center', va='center',size = 13)
    # cbar.ax.text(3.2, -54, 'ΔRMSE < 0', ha='center', va='center',size = 13)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels,size = 20)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels,size = 20)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-60, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar,ax


# def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
#                      textcolors=("black", "black"),
#                      threshold=None, **textkw):

def annotate_heatmap(im, data=None, valfmt="{:.6f}",
                     textcolors=("black", "black"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # data = abs(data)

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            # text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            text = im.axes.text(j, i, np.round(data[i, j],2), **kw)

            texts.append(text)

    return texts




# brc_30 = pd.read_csv('/stu01/shijh21/liuz/sis/bias_rmse_corr_30min.csv',index_col=0)
brc_30 = pd.read_csv('/stu01/shijh21/liuz/sis/bias_rmse_corr_day.csv',index_col=0)




bias = np.array([
    [brc_30['lh_igbp_bias'].values[0],brc_30['lh_pft_bias'].values[0],brc_30['lh_pc_bias'].values[0],],
    [brc_30['h_igbp_bias'].values[0],brc_30['h_pft_bias'].values[0],brc_30['h_pc_bias'].values[0],],
    
    [brc_30['r_igbp_bias'].values[0],brc_30['r_pft_bias'].values[0],brc_30['r_pc_bias'].values[0],],
    [brc_30['gpp_igbp_bias'].values[0],brc_30['gpp_pft_bias'].values[0],brc_30['gpp_pc_bias'].values[0],],
    [brc_30['ustar_igbp_bias'].values[0],brc_30['ustar_pft_bias'].values[0],brc_30['ustar_pc_bias'].values[0],],
    ])
rmse = np.array([
    [brc_30['lh_igbp_rmse'].values[0],brc_30['lh_pft_rmse'].values[0],brc_30['lh_pc_rmse'].values[0],],
    [brc_30['h_igbp_rmse'].values[0],brc_30['h_pft_rmse'].values[0],brc_30['h_pc_rmse'].values[0],],
    
    [brc_30['r_igbp_rmse'].values[0],brc_30['r_pft_rmse'].values[0],brc_30['r_pc_rmse'].values[0],],
    [brc_30['gpp_igbp_rmse'].values[0],brc_30['gpp_pft_rmse'].values[0],brc_30['gpp_pc_rmse'].values[0],],
    [brc_30['ustar_igbp_rmse'].values[0],brc_30['ustar_pft_rmse'].values[0],brc_30['ustar_pc_rmse'].values[0],],
    ])

corr = np.array([
    [brc_30['lh_igbp_corr'].values[0],brc_30['lh_pft_corr'].values[0],brc_30['lh_pc_corr'].values[0],],
    [brc_30['h_igbp_corr'].values[0],brc_30['h_pft_corr'].values[0],brc_30['h_pc_corr'].values[0],],
    
    [brc_30['r_igbp_corr'].values[0],brc_30['r_pft_corr'].values[0],brc_30['r_pc_corr'].values[0],],
    [brc_30['gpp_igbp_corr'].values[0],brc_30['gpp_pft_corr'].values[0],brc_30['gpp_pc_corr'].values[0],],
    [brc_30['ustar_igbp_corr'].values[0],brc_30['ustar_pft_corr'].values[0],brc_30['ustar_pc_corr'].values[0],],
    ])
col_labels_bias = ['Bias',]
col_labels_rmse = ['RMSE',]
col_labels_corr = ['Corr',]

col_labels = ['IBGP','PFT','PC']

row_labels = ['LE','H','Rn','GPP','Ustar']



fig, ax = plt.subplots(figsize=(8, 13))

# fig,ax = plt.subplots(1,3,figsize = (15,13),
#                             sharex=True,sharey='row')



n_colors = 4096  # Number of discrete colors in the colormap
cmap = mcolors.LinearSegmentedColormap.from_list("custom_diverging",
                                                  ['#00bfff','white','#ff8900'],
                                                    N=n_colors)


    
# im, cbar,axx = heatmap(bias, row_labels,col_labels, ax=ax,
#                    cmap=cmap, aspect='auto',cbarlabel="BIAS")
# fig.suptitle("Bias", x=0.5, y=1.05, fontsize=40, color='red')
# texts = annotate_heatmap(im, valfmt="{x:.1f}",fontsize = 30)



# im, cbar,axx = heatmap_rmse(rmse, row_labels,col_labels, ax=ax,
#                    cmap='GnBu', aspect='auto',cbarlabel="RMSE")
# texts = annotate_heatmap(im, valfmt="{x:.1f}",fontsize = 30)
# fig.suptitle("RMSE", x=0.5, y=1.05, fontsize=40, color='red')


im, cbar,axx = heatmap_corr(corr, row_labels,col_labels, ax=ax,
                   cmap='GnBu', aspect='auto',cbarlabel="Corr")
texts = annotate_heatmap(im, valfmt="{x:.1f}",fontsize = 30)
fig.suptitle("Corr", x=0.5, y=1.05, fontsize=40, color='red')


fig.tight_layout()
plt.show()

































