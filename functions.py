import cartopy
import matplotlib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from datetime import timedelta

def get_analog_forecast_times(ds, station=None, station_index=0,
                              time=None, time_index=None,
                              lead_time=3600 * 18, lead_time_index=None, target_station_index=0):
    
    # Sanity check
    assert station is None, 'Indexing with station values is not yet supported. Please provide station_index instead.'
    assert station_index is not None, 'station_index is required!'
    
    if not ((time is None) ^ (time_index is None)):
        raise Exception('Please provide either a time or a time_index!')
    
    if not ((lead_time is None) ^ (lead_time_index is None)):
        raise Exception('Please provide either a lead_time or a lead_time_index!')
    
    # Index the station
    da = ds['similarity_time_index'].isel(num_stations=target_station_index)
    
    # Index the time
    da = da.isel(num_times=time_index) if time is None else da.sel(num_times=time)
    
    # Index the lead time
    da = da.isel(num_flts=lead_time_index) if time is None else da.sel(num_flts=lead_time)
    
    # Convert types
    da = da.data
    da = da[~np.isnan(da)]
    da = da.astype(int)
    
    # Get times
    da = np.array([pd.to_datetime(e) for e in ds['search_times'].data[da]])
    
    return da

def plot_analog_forecasts(obs_ds, ds, analog_times, ds_obs,
                          analog_lead_time=3600*18, target_lead_time=3600*18,
                          out_file=None, vmin=None, vmax=None, return_range=False, sort_values=True):
    
    assert len(analog_times) == 1 + 21, 'There must be 1 (target time) and 21 (ensemble member times)'
    
    ################
    # Prepare data #
    ################
    
    # Sort analog times based on observations
    if sort_values:
        obs_values = []
        
        # Do not sort the first value because it is the target time
        for analog_time in analog_times[1:]:
            obs_time = pd.to_datetime(analog_time) + timedelta(seconds=analog_lead_time)
            obs_values.append(ds_obs['Data'].sel(num_parameters='dw_solar', num_times=obs_time).squeeze('num_stations').item())

        analog_times[1:] = analog_times[1:][np.argsort(obs_values)]

    all_values = []

    for analog_time in analog_times:
        all_values.append(ds['Data'].sel(
            num_times=analog_time,
            num_parameters='260087_0_surface_dswrf',
            num_flts=target_lead_time).data)

    all_values = np.stack(all_values)
    
    if vmin is None:
        vmin = all_values.min()
        
    if vmax is None:
        vmax = all_values.max()
    
    if return_range:
        return vmin, vmax
    
    #########
    # Plots #
    #########

    fig = plt.figure(figsize=(16.7, 16))
    gs = matplotlib.gridspec.GridSpec(5, 5)

    crs_visual = cartopy.crs.PlateCarree()

    counter = 0

    for i in range(5):
        for j in range(5):

            if i==0 and j==0:
                # Add the target forecast
                ax = fig.add_subplot(gs[:2, :2], projection=crs_visual)
                pt_size = [30, 60]

            elif i < 2 and j < 2:
                # No action
                continue

            else:
                # Add analog forecasts
                ax = fig.add_subplot(gs[i, j], projection=crs_visual)
                pt_size = [8, 20]

            im = ax.scatter(ds['Xs'], ds['Ys'], c=all_values[counter], s=pt_size[0],
                            vmin=vmin, vmax=vmax, zorder=6, cmap='Spectral_r')
            
            ax.scatter(obs_ds['Xs'].item(), obs_ds['Ys'].item(), marker='*',
                       facecolors='red', edgecolors='black', s=pt_size[1], label='SURFRAD', zorder=7)
            
            if counter == 0:
                ax.legend(loc='upper right', fancybox=False, facecolor='white', framealpha=1, edgecolor='black')

            ax.set_xlim(-81.5, -74.5)
            ax.set_ylim(37.5, 44.5)
        
            gl = ax.gridlines(linestyle='dashed', color='lightgrey', draw_labels=True, zorder=7)
            gl.top_labels = False
            gl.right_labels = False
            gl.bottom_labels = True if i==4 else False
            gl.left_labels = True if j==0 else False

            alpha = 0.4
            ax.add_feature(cartopy.feature.OCEAN, linewidth=1, zorder=5, alpha=alpha)
            ax.add_feature(cartopy.feature.LAKES, linewidth=1, zorder=5, alpha=alpha)
            ax.add_feature(cartopy.feature.STATES, linewidth=1, zorder=8)
        
            obs_time = pd.to_datetime(analog_times[counter]) + timedelta(seconds=analog_lead_time)
            obs = ds_obs['Data'].sel(num_parameters='dw_solar', num_times=obs_time).squeeze('num_stations').item()
            figure_text = '({}) {}: {:0.2f} $W/m^2$'.format('Target' if counter==0 else counter, obs_time.strftime('%Y/%m/%d'), obs)

            t = ax.text(0.03, 0.97, figure_text, horizontalalignment='left',
                        verticalalignment='top', transform=ax.transAxes, zorder=9)

            t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))

            counter += 1

    fig.subplots_adjust(left=0.025, right=0.95, bottom=0.02, top=0.99, hspace=0.012, wspace=0.008)

    cb_ax = fig.add_axes([0.95, 0.02, 0.01, 0.97])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label('Solar irradiance [$W/m^2$]')
    
    if out_file is None:
        fig.show()
    else:
        plt.savefig(out_file, dpi=100)
        plt.close()
        
def find_closest_index(f1, f2, i, n, include_self=False):
    
    # Normalize
    feature1 = (f1 - np.nanmin(f1)) / (np.nanmax(f1) - np.nanmin(f1))
    feature2 = (f2 - np.nanmin(f2)) / (np.nanmax(f2) - np.nanmin(f2))
    
    # Distance
    dist = np.abs(feature1[i] - feature1) + np.abs(feature2[i] - feature2)
    rank = np.argsort(dist)

    if include_self:
        return rank[:n]
    else:
        return rank[1:(n+1)]
    
def plot_cluster(datetimes, ds_fcsts, ds_obs, nbs, axes, vmin=None, vmax=None, label_start=None, target_lead_time=3600*18):
    fcst_times = np.array([datetimes[i] for i in nbs])
    fcst_cluster = ds_fcsts['Data'].sel(num_times=fcst_times, num_flts=target_lead_time, num_parameters='260087_0_surface_dswrf').data
    obs_cluster = ds_obs['Data'].sel(num_times=fcst_times + timedelta(seconds=target_lead_time), num_parameters='dw_solar').squeeze('num_stations').data
    x, y = ds_fcsts['Xs'].data, ds_fcsts['Ys']
    
    if vmin is None and vmax is None:
        print(fcst_cluster.min(), fcst_cluster.max())
    
    if vmin is None:
        vmin = fcst_cluster.min()
    
    if vmax is None:
        vmax = fcst_cluster.max()
    
    if label_start is None:
        label_start = 1

    i = 0

    for ax in axes:

        gl = ax.gridlines(linestyle='dashed', color='lightgrey', draw_labels=True, zorder=7)
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = False
        gl.left_labels = False

        alpha = 0.4
        ax.add_feature(cartopy.feature.OCEAN, linewidth=1, zorder=5, alpha=alpha)
        ax.add_feature(cartopy.feature.LAKES, linewidth=1, zorder=5, alpha=alpha)
        ax.add_feature(cartopy.feature.STATES, linewidth=1, zorder=8)

        im = ax.scatter(x, y, c=fcst_cluster[i], cmap='Spectral_r', s=5, edgecolor='none', vmin=vmin, vmax=vmax, zorder=7)
        ax.scatter(x[1221], y[1221], c='black', s=40, marker='*', zorder=8)
        ax.text(0.01, 0.01, '({})'.format(label_start + i), verticalalignment='bottom',
                horizontalalignment='left', transform=ax.transAxes, zorder=9)

        i += 1

        ax.set_xticks([])
        ax.set_yticks([])

    return im

def get_attribution(*datetimes, best_feature_index=None, use_baseline=False, n_samples=50):
    
    # Get sample inputs
    inputs = get_inputs(ds_fcsts, datetimes, 1221, 39, 39, grid)

    # Get the embeddings
    embeddings = embedding_net(inputs).detach().numpy()
    
    if best_feature_index is None:

        # Calculate the most helpful feature index
        dif1 = np.abs(embeddings[1] - embeddings[0])
        dif2 = np.abs(embeddings[2] - embeddings[0])
        best_feature_index = np.argsort(np.abs(dif1 - dif2))[-1]

    # Calculate gradients with noise
    inputs.requires_grad = True
    embedding_net.zero_grad()

    ig = IntegratedGradients(embedding_net)
    nt = NoiseTunnel(ig)
    
    if use_baseline:
        baselines = multi_layer_blur(inputs.detach().numpy(), sigma=2)
        baselines = torch.from_numpy(baselines)
        attribution = nt.attribute(inputs, target=int(best_feature_index), nt_type='smoothgrad_sq', stdevs=0.5, n_samples=n_samples, baselines=baselines)
    else:
        attribution = nt.attribute(inputs, target=int(best_feature_index), nt_type='smoothgrad_sq', stdevs=0.5, n_samples=n_samples)

    # Max across lead times
    attribution = attribution.abs().detach().numpy().max(4)
    
    # Calculate parameter index
    parameter_index = np.where(ds_fcsts['ParameterNames'] == '260087_0_surface_dswrf')[0][0]
    
    # Prepare return
    inputs = inputs.detach().numpy()[:, parameter_index, :, :, 1]
    attribution = attribution[:, parameter_index]
    
    return inputs, attribution, best_feature_index


def plot_feature(feature_index, center1, center2, center3, n_nbs, embeddings, datetimes, fcst_ds, obs_ds):
    
    obs = obs_ds['Data'].squeeze('num_stations').sel(num_parameters='dw_solar', num_times=datetimes + timedelta(seconds=3600 * 18)).data

    center1_nbs = find_closest_index(embeddings[feature_index], obs, center1, n_nbs, include_self=True)
    center2_nbs = find_closest_index(embeddings[feature_index], obs, center2, n_nbs, include_self=True)
    center3_nbs = find_closest_index(embeddings[feature_index], obs, center3, n_nbs, include_self=True)

    #### Visualization ####

    fig = plt.figure(figsize=(22, 15))
    gs = matplotlib.gridspec.GridSpec(6, 8)

    ax_left_top = fig.add_subplot(gs[:, :3])

    ax_left_top.grid(c='lightgrey')

    ax_left_top.scatter(embeddings[feature_index], obs, s=6, c='lightgrey', label='Search')

    ax_left_top.scatter(embeddings[feature_index, center1_nbs], obs[center1_nbs], s=20, c='lightgreen', label='Cluster 1 (2~11)')
    ax_left_top.scatter(embeddings[feature_index, center2_nbs], obs[center2_nbs], s=20, c='red', label='Cluster 2 (12~21)')
    ax_left_top.scatter(embeddings[feature_index, center3_nbs], obs[center3_nbs], s=20, c='lightblue', label='Cluster 3 (22~31)')

    ax_left_top.scatter(embeddings[feature_index, center1], obs[center1], s=40, c='darkgreen', marker='*')
    ax_left_top.scatter(embeddings[feature_index, center2], obs[center2], s=40, c='darkred', marker='*')
    ax_left_top.scatter(embeddings[feature_index, center3], obs[center3], s=40, c='blue', marker='*', label='Cluster center')

    ax_left_top.set_ylabel('Solar irradiance [$W/m^2$]')
    ax_left_top.set_xlabel('Latent feature #{}'.format(feature_index))

    lgd = ax_left_top.legend(loc='lower left', fancybox=False, edgecolor='black', facecolor='white', framealpha=1, ncol=1, handletextpad=0.1)

    lgd.legendHandles[0].set_sizes([20])
    lgd.legendHandles[4].set_sizes([20])
    lgd.legendHandles[4].set_color(['black'])

    ax_left_top.text(0.99, 0.01, '(1)', verticalalignment='bottom', horizontalalignment='right', transform=ax_left_top.transAxes)

    vmin, vmax = 0, 1050

    axes = [fig.add_subplot(gs[j, i], projection=cartopy.crs.PlateCarree())
            for j in range(2) for i in range(3, 8)]
    plot_cluster(datetimes, fcst_ds, obs_ds, center1_nbs, axes, vmin=vmin, vmax=vmax, label_start=2)

    axes = [fig.add_subplot(gs[j, i], projection=cartopy.crs.PlateCarree())
            for j in range(2, 4) for i in range(3, 8)]
    plot_cluster(datetimes, fcst_ds, obs_ds, center2_nbs, axes, vmin=vmin, vmax=vmax, label_start=12)

    axes = [fig.add_subplot(gs[j, i], projection=cartopy.crs.PlateCarree())
            for j in range(4, 6) for i in range(3, 8)]
    im = plot_cluster(datetimes, fcst_ds, obs_ds, center3_nbs, axes, vmin=vmin, vmax=vmax, label_start=22)

    fig.subplots_adjust(left=0.068, right=0.92, bottom=0.06, top=0.99, wspace=0.04, hspace=0.08)
    cbar_ax = fig.add_axes([0.925, 0.06, 0.01, 0.93])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Solar irradiance [$W/m^2$]')
    
    anchor_left = 0.388
    box_height = 0.309
    box_width = 0.532
    fig.patches.extend([plt.Rectangle((anchor_left,0.685), box_width, box_height, fill=False, color='lightgreen',
                                      linewidth=5, zorder=1000, transform=fig.transFigure, figure=fig)])
    fig.patches.extend([plt.Rectangle((anchor_left,0.370), box_width, box_height, fill=False, color='red',
                                      linewidth=5, zorder=1000, transform=fig.transFigure, figure=fig)])
    fig.patches.extend([plt.Rectangle((anchor_left,0.056), box_width, box_height, fill=False, color='lightblue',
                                      linewidth=5, zorder=1000, transform=fig.transFigure, figure=fig)])

    fig.show()
    