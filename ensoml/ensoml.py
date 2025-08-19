"""
This code generates Figures 1 through 8 of the "Machine learning 
exploration of the El Nino prediction skills from the deep ocean
temperature over the Western Tropical Pacific" Paper. Using GODAS Data
"""

# Basic imports
import os
import time
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats
import plotly.graph_objects as go
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
import sklearn.metrics, sklearn.linear_model

# Keep these imports (maybe for future use)
from neuralforecast import NeuralForecast  # noqa: F401
from neuralforecast.models import LSTM, NHITS, Autoformer, Informer 

# Optional: seaborn 
import seaborn as sns 


# Global style (Arial, bold)
mpl.rcParams.update({
    "font.family": "Arial",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": 16,
    "axes.labelsize": 14,
})

DEG = u"\N{DEGREE SIGN}"


def lon_label(x, _):
    if np.isnan(x):
        return ""
    x = float(x)
    if x <= 180:
        return f"{int(x)}{DEG}E"
    w = int(360 - x)
    return f"{w}{DEG}W" if w != 0 else f"180{DEG}"


def lat_label(y, _):
    if np.isnan(y):
        return ""
    y = int(round(y))
    if y > 0:
        return f"{y}{DEG}N"
    if y < 0:
        return f"{-y}{DEG}S"
    return f"0{DEG}"


def _apply_common_axes_format(ax):
    ax.set_xticks([120, 150, 180, 210, 240, 270])
    ax.xaxis.set_major_formatter(FuncFormatter(lon_label))
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.yaxis.set_major_formatter(FuncFormatter(lat_label))
    ax.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.2)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)


# Figure 2
def generate_figure2_like_paper():
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    ds = xr.open_dataset("data/godasClimatologyData_5m.nc")
    months = ['1998-01-16', '1998-02-16', '1998-03-16',
              '1998-04-16', '1998-05-16', '1998-06-16']
    titles = ['January 1998', 'February 1998', 'March 1998',
              'April 1998', 'May 1998', 'June 1998']
    nino_lat_range = (-5, 5)
    nino_lon_range = (190, 240)

    fig = plt.figure(figsize=(12, 13))
    gs = fig.add_gridspec(
        nrows=len(months), ncols=2, width_ratios=[1, 0.045], wspace=0.06, hspace=0.32
    )
    axs = [fig.add_subplot(gs[i, 0]) for i in range(len(months))]

    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad('#ffffff')
    vmin, vmax = -5, 5

    mappable = None
    for i, month in enumerate(months):
        temp = ds['deepTemp'].sel(time=month).squeeze()
        temp = temp.sel(lat=slice(-20, 20), lon=slice(120, 280))
        temp = temp.where(~np.isnan(temp), other=np.nan)

        pcm = axs[i].pcolormesh(
            temp['lon'], temp['lat'], temp, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto'
        )
        mappable = pcm

        axs[i].set_xticks([120, 150, 180, 210, 240, 270])
        axs[i].xaxis.set_major_formatter(FuncFormatter(lon_label))
        axs[i].set_yticks([-20, -10, 0, 10, 20])
        axs[i].yaxis.set_major_formatter(FuncFormatter(lat_label))
        axs[i].tick_params(axis='both', labelsize=12, length=6, width=1.2)
        axs[i].set_ylabel('Latitude', fontsize=13)

        axs[i].text(
            0.01, 0.97, titles[i], transform=axs[i].transAxes,
            ha='left', va='top', fontsize=18, fontweight='bold'
        )

        rect = patches.Rectangle(
            (nino_lon_range[0], nino_lat_range[0]),
            nino_lon_range[1] - nino_lon_range[0],
            nino_lat_range[1] - nino_lat_range[0],
            linewidth=2.0, edgecolor='red', facecolor='none'
        )
        axs[i].add_patch(rect)

        if i < len(months) - 1:
            axs[i].annotate('↓', xy=(250, -17), xytext=(250, -18),
                            fontsize=18, ha='center', va='center')

    axs[-1].set_xlabel('Longitude', fontsize=13)
    fig.suptitle("Predictors and Predictands at 5 meter Depth", fontsize=35)

    cax = fig.add_subplot(gs[:, 1])
    cbar = fig.colorbar(mappable, cax=cax, orientation='vertical', extend='both')
    cbar.set_label('°C', fontsize=13)
    cbar.ax.tick_params(labelsize=12, width=1.2, length=6)

    fig.subplots_adjust(top=0.92, left=0.08, right=0.86, bottom=0.07)
    plt.savefig("figures/figure2_surface_stack.png", dpi=400, bbox_inches='tight')
    plt.show()
    ds.close()


# Figure 3
def generate_figure3_like_paper():
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    ds = xr.open_dataset("data/godasClimatologyData_115m.nc")
    months = ['1998-01-16', '1998-02-16', '1998-03-16',
              '1998-04-16', '1998-05-16', '1998-06-16']
    titles = ['January 1998', 'February 1998', 'March 1998',
              'April 1998', 'May 1998', 'June 1998']
    nino_lat_range = (-10, 10)
    nino_lon_range = (175, 225)

    fig = plt.figure(figsize=(12, 13))
    gs = fig.add_gridspec(
        nrows=len(months), ncols=2, width_ratios=[1, 0.045], wspace=0.06, hspace=0.32
    )
    axs = [fig.add_subplot(gs[i, 0]) for i in range(len(months))]

    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad('white')  # land/NaN color (kept)
    vmin, vmax = -8, 8

    mappable = None
    import matplotlib.patheffects as pe

    for i, month in enumerate(months):
        temp = ds['deepTemp'].sel(time=month).squeeze()
        temp = temp.sel(lat=slice(-20, 20), lon=slice(120, 280))
        temp = temp.where(~np.isnan(temp), other=np.nan)

        pcm = axs[i].pcolormesh(
            temp['lon'], temp['lat'], temp, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto'
        )
        mappable = pcm

        axs[i].set_xticks([120, 150, 180, 210, 240, 270])
        axs[i].xaxis.set_major_formatter(FuncFormatter(lon_label))
        axs[i].set_yticks([-20, -10, 0, 10, 20])
        axs[i].yaxis.set_major_formatter(FuncFormatter(lat_label))
        axs[i].tick_params(axis='both', labelsize=12, length=6, width=1.2)
        axs[i].set_ylabel('Latitude', fontsize=13)

        axs[i].text(
            0.01, 0.97, titles[i],
            transform=axs[i].transAxes, ha='left', va='top',
            fontsize=18, fontweight='bold', color='white',
            path_effects=[pe.withStroke(linewidth=3, foreground='#1f2937')]
        )

        rect = patches.Rectangle(
            (nino_lon_range[0], nino_lat_range[0]),
            nino_lon_range[1] - nino_lon_range[0],
            nino_lat_range[1] - nino_lat_range[0],
            linewidth=2.0, edgecolor='red', facecolor='none'
        )
        axs[i].add_patch(rect)

        if i < len(months) - 1:
            axs[i].annotate('↓', xy=(250, -17), xytext=(250, -18),
                            fontsize=18, ha='center', va='center')

    axs[-1].set_xlabel('Longitude', fontsize=13)
    fig.suptitle("Predictors and Predictands at 115 meter Depth", fontsize=35)

    cax = fig.add_subplot(gs[:, 1])
    cbar = fig.colorbar(mappable, cax=cax, orientation='vertical', extend='both')
    cbar.set_label('°C', fontsize=13)
    cbar.ax.tick_params(labelsize=12, width=1.2, length=6)

    fig.subplots_adjust(top=0.92, left=0.08, right=0.86, bottom=0.07)
    plt.savefig("figures/figure3_deep_stack.png", dpi=400, bbox_inches='tight')
    plt.show()
    ds.close()


# EnsoModel class
class EnsoModel:
    def __init__(self):
        self.dataset_begin = '1980-01-01'
        self.dataset_end = '2017-12-31'
        # Default train / validation date ranges
        self.start_stop_list_train = ['1980-01-01', '1995-12-31']
        self.start_stop_list_val = ['1997-01-01', '2006-12-31']
        self.save_time_series_plots = True
        for x in ['stats', 'figures']:
            if not os.path.isdir(x):
                os.mkdir(x)

    def data_assembly(self, date_list, path_list, nan_mask, lead_time):
        """Return (X, y) for a given date window, predictor file, and lead."""
        start_date, end_date = date_list[0], date_list[1]
        godas_path, anomaly_path = path_list[0], path_list[1]

        ds = xr.open_dataset(godas_path)
        deepTemp = ds['deepTemp'].sel(time=slice(start_date, end_date))
        deepTemp_reshape = deepTemp.values.reshape(deepTemp.shape[0], -1)
        X = deepTemp_reshape[:, nan_mask]
        ds.close()

        start_date_plus_lead = pd.to_datetime(start_date) + pd.DateOffset(months=lead_time)
        end_date_plus_lead = pd.to_datetime(end_date) + pd.DateOffset(months=lead_time)

        with open(anomaly_path) as f:
            line = f.readline()
            enso_vals = []
            while line:
                yearly_enso_vals = map(float, line.split()[1:])
                enso_vals.extend(yearly_enso_vals)
                line = f.readline()
        enso_vals = pd.Series(enso_vals)
        enso_vals.index = pd.date_range(self.dataset_begin, freq='MS', periods=len(enso_vals))
        enso_vals.index = pd.to_datetime(enso_vals.index)
        y = enso_vals[slice(start_date_plus_lead, end_date_plus_lead)]
        return X, y

    def save_figs(self, y_true, predictions, predictor_depth, predictand_depth, lead, title):
        """Save a quick line plot of ground truth vs predictions."""
        fig_png = f'predictor_{predictor_depth}m_predictand_{predictand_depth}m_{lead}month_Predictions.png'
        fig_pdf = f'predictor_{predictor_depth}m_predictand_{predictand_depth}m_{lead}month_Predictions.pdf'

        predictions = pd.Series(predictions, index=y_true.index).sort_index()
        y_true = y_true.sort_index()

        plt.plot(y_true, label='Ground Truth')
        plt.plot(predictions, '--', label='LR Predictions')
        plt.legend(loc='best')
        plt.title(title)
        plt.ylabel(f'{predictand_depth}m Temperature Anomalies')
        plt.xlabel('Time (Months)')
        for f in [fig_png, fig_pdf]:
            plt.savefig(os.path.join('figures', f))
        plt.close()

    def obtain_dir(self, predictor_depth=5, predictand_depth=5):
        """Return predictor NetCDF and target anomalies path."""
        return [
            os.path.join("data", f"godasClimatologyData_{predictor_depth}m.nc"),
            os.path.join("data", f"movingAverageAnomalies{predictand_depth}m.txt"),
        ]

    def corr_rmse_fcn(self, y_true, predictions):
        """Compute correlation and MSE (kept as-is)."""
        return (
            scipy.stats.pearsonr(y_true, predictions)[0],
            sklearn.metrics.mean_squared_error(y_true, predictions),
        )

    def generate_figure1_plotly_volume_like(self):
        depth_vec = [5, 35, 65, 95, 125, 155, 165]
        target_dates = {
            "Tropical Pacific Anomalies, January 1998": "1998-01-16",
            "Tropical Pacific Anomalies, January 2000": "2000-01-16",
        }
        lat_range = slice(-20, 20)
        lon_range = slice(120, 280)

        n = len(depth_vec)
        z_positions = np.arange(n, dtype=float)

        # Ends more opaque, middle more transparent – set to 1 to keep opaque, 0 to make transparent
        end_alpha, mid_alpha = 1, 1
        t = np.linspace(0, np.pi, n)
        opacities = mid_alpha + (end_alpha - mid_alpha) * (np.cos(t) ** 2)

        for title, date_str in target_dates.items():
            traces = []
            for k, depth in enumerate(depth_vec):
                path = os.path.join("data", f"godasClimatologyData_{depth}m.nc")
                if not os.path.exists(path):
                    print(f"Missing {path}")
                    continue

                ds = xr.open_dataset(path)
                try:
                    sel = ds.sel(time=np.datetime64(date_str), method="nearest")
                    temp = sel["deepTemp"].sel(lat=lat_range, lon=lon_range)

                    lats = temp["lat"].values
                    lons = temp["lon"].values
                    vals = temp.values

                    X, Y = np.meshgrid(lons, lats)

                    traces.append(
                        go.Surface(
                            x=X, y=Y, z=np.full_like(vals, z_positions[k]),
                            surfacecolor=vals,
                            colorscale="Jet", cmin=-5, cmax=5,
                            opacity=float(opacities[k]),
                            showscale=(k == 0),
                            colorbar=dict(title="Anomaly (°C)", thickness=18, len=0.85),
                            lighting=dict(ambient=0.95, diffuse=0.05, specular=0.0,
                                          roughness=1.0, fresnel=0.0),
                            lightposition=dict(x=0, y=0, z=1000),
                            showlegend=False,
                        )
                    )
                finally:
                    ds.close()

            fig = go.Figure(traces)
            fig.update_layout(
                title=title,
                template="plotly_white",
                margin=dict(l=10, r=10, t=40, b=10),
                font=dict(family="Arial", size=14),
                scene_aspectmode="cube",
                scene_camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.35, y=1.2, z=0.8)),
                scene=dict(
                    xaxis=dict(
                        title="Longitude",
                        tickvals=[120, 150, 180, 210, 240, 270],
                        ticktext=["120°E", "150°E", "180°", "150°W", "120°W", "90°W"],
                        autorange=True,  # kept as-is (not reversed) to preserve behavior
                    ),
                    yaxis=dict(
                        title="Latitude",
                        tickvals=[-20, -10, 0, 10, 20],
                        ticktext=["20°S", "10°S", "0°", "10°N", "20°N"],
                        autorange="reversed",  # kept as-is (not reversed) to preserve behavior
                    ),
                    zaxis=dict(
                        title="Depth (m)",
                        tickvals=z_positions.tolist(),
                        ticktext=[f"{d} m" for d in depth_vec],
                        autorange="reversed",  # depth increases downward
                    ),
                ),
            )
            fig.show()


# Extra helpers 
def _smart_legend(ax, y_true, y_pred):
    import pandas as pd
    df = pd.DataFrame({
        "y": y_true.sort_index(),
        "p": pd.Series(y_pred, index=y_true.index).sort_index()
    }).dropna()
    n = len(df)
    thirds = [slice(0, n // 3), slice(n // 3, 2 * n // 3), slice(2 * n // 3, n)]
    spreads = []
    for s in thirds:
        sub = df.iloc[s]
        spreads.append((sub.max() - sub.min()).sum())
    i = int(np.argmin(spreads))
    locs = [("upper left", (0.02, 0.98)),
            ("upper center", (0.5, 0.98)),
            ("upper right", (0.98, 0.98))]
    loc, anchor = locs[i]
    ax.legend(loc=loc, bbox_to_anchor=anchor, frameon=True, framealpha=0.9,
              borderpad=0.4, fontsize=10)


def _style_ts_ax(ax):
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.tick_params(labelsize=11, length=5, width=1.1)
    for sp in ax.spines.values():
        sp.set_linewidth(1.1)


def _panel_label(ax, label):
    ax.text(0.01, 0.97, label, transform=ax.transAxes,
            ha="left", va="top", fontsize=12, fontweight="bold")


def _legend_outside_left(ax):
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02),
              frameon=True, framealpha=0.95, borderpad=0.3, fontsize=10)


def _panel_label_outside(ax, label):
    ax.text(-0.075, 1.02, label, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=12, fontweight='bold')


# EnsoLinear Model (hold figures 4–6)
class EnsoLinearModel(EnsoModel):
    def __init__(self):
        super(EnsoLinearModel, self).__init__()

    def run_all_scenarios(self, predictand_depth=5, use_deep_target=False):
        """Sweep depths × leads and compute corr/RMSE matrices."""
        month_vec = np.arange(1, 19)  # note: comment kept as in your code
        depth_vec = np.arange(5, 205, 10)  # 5–195 m
        corr_mat = np.zeros((len(month_vec), len(depth_vec)))
        rmse_mat = np.zeros((len(month_vec), len(depth_vec)))

        for mm, depth_sel in enumerate(depth_vec):
            directory_list = self.obtain_dir(
                predictor_depth=depth_sel, predictand_depth=predictand_depth
            )
            ds = xr.open_dataset(directory_list[0])
            deepTemp = ds['deepTemp']
            deepTemp_reshape = deepTemp.values.reshape(deepTemp.shape[0], -1)
            non_nan_columns = ~np.isnan(deepTemp_reshape).any(axis=0)
            ds.close()

            for nn, lead in enumerate(month_vec):
                X_train, y_train = self.data_assembly(
                    self.start_stop_list_train, directory_list, non_nan_columns, lead
                )
                X_val, y_val = self.data_assembly(
                    self.start_stop_list_val, directory_list, non_nan_columns, lead
                )

                if use_deep_target:
                    with open(directory_list[1]) as f:
                        enso_vals = []
                        line = f.readline()
                        while line:
                            enso_vals.extend(map(float, line.split()[1:]))
                            line = f.readline()
                    y_series = pd.Series(enso_vals)
                    y_series.index = pd.date_range(self.dataset_begin, freq='MS', periods=len(y_series))

                    start_val = pd.to_datetime(self.start_stop_list_val[0]) + pd.DateOffset(months=lead)
                    end_val = pd.to_datetime(self.start_stop_list_val[1]) + pd.DateOffset(months=lead)
                    y_val = y_series[start_val:end_val]
                    y_val = y_val[:len(X_val)]

                regr = sklearn.linear_model.LinearRegression()
                regr.fit(X_train, y_train)
                y_predicted = regr.predict(X_val)

                corr, rmse = self.corr_rmse_fcn(y_val, y_predicted)
                corr_mat[nn, mm] = corr
                rmse_mat[nn, mm] = rmse

                if self.save_time_series_plots:
                    fig_title = (
                        f'Predicted vs True\nLead={lead}, Depth={depth_sel}m\n'
                        f'Corr={corr:.2f}, RMSE={rmse:.2f}'
                    )
                    self.save_figs(y_val, y_predicted, depth_sel, predictand_depth, lead, fig_title)

        return corr_mat, rmse_mat, month_vec, depth_vec

    def run_single_scenario(self, predictor_depth=5, predictand_depth=5):
        month_vec = np.arange(1, 15)
        directory_list = self.obtain_dir(predictor_depth=predictor_depth)
        corr_vals = np.zeros(len(month_vec))
        rmse_vals = np.zeros(len(month_vec))
        data_point_count = np.zeros(len(month_vec))

        ds = xr.open_dataset(directory_list[0])
        deepTemp = ds['deepTemp']
        deepTemp_reshape = deepTemp.values.reshape(deepTemp.shape[0], -1)
        non_nan_columns = ~np.isnan(deepTemp_reshape).any(axis=0)
        ds.close()

        for idx in range(len(month_vec)):
            X_train, y_train = self.data_assembly(
                self.start_stop_list_train, directory_list, non_nan_columns, month_vec[idx]
            )
            X_val, y_val = self.data_assembly(
                self.start_stop_list_val, directory_list, non_nan_columns, month_vec[idx]
            )
            regr = sklearn.linear_model.LinearRegression()
            regr.fit(X_train, y_train)
            y_predicted = regr.predict(X_val)
            corr, rmse = self.corr_rmse_fcn(y_val, y_predicted)
            corr_vals[idx] = corr
            rmse_vals[idx] = rmse
            data_point_count[idx] = len(y_val)

            if self.save_time_series_plots:
                fig_title1 = f'Predicted and True Temperature Anomalies at {month_vec[idx]} MonthLead Time \n'
                fig_title2 = 'Corr : %.2f, RMSE : %.2f %(corr, rmse)'
                fig_title = fig_title1 + fig_title2
                self.save_figs(y_val, y_predicted, predictor_depth, predictand_depth, month_vec[idx], fig_title)

        df = pd.concat([
            pd.DataFrame({'Values': corr_vals, 'Metric': 'Pearson Correlation'}),
            pd.DataFrame({'Values': rmse_vals, 'Metric': 'Root Mean Square Error'})
        ])
        df['Time Lead [months'] = np.concatenate((month_vec, month_vec))
        df['Predictor Depth [meters]'] = predictor_depth
        df['Predictand Depth [meters]'] = predictand_depth
        df['Number of Data Points'] = np.concatenate((data_point_count, data_point_count))
        df.to_csv(os.path.join('stats', 'lr_results.csv'))
        return

    # Figures 4–6 
    def generate_figure4(self):
        import matplotlib.ticker as mticker

        predictor_depth = 5
        predictand_depth = 5
        lead_times = [1, 3, 6, 9]
        directory_list = self.obtain_dir(
            predictor_depth=predictor_depth, predictand_depth=predictand_depth
        )

        ds = xr.open_dataset(directory_list[0])
        deepTemp = ds['deepTemp'].values.reshape(ds['deepTemp'].shape[0], -1)
        non_nan_columns = ~np.isnan(deepTemp).any(axis=0)
        ds.close()

        fig, axs = plt.subplots(len(lead_times), 1, figsize=(12, 10), sharex=True, constrained_layout=True)
        labels = ["(a)", "(b)", "(c)", "(d)"]
        y_min, y_max = -2.5, 2.5

        for i, lead in enumerate(lead_times):
            X_tr, y_tr = self.data_assembly(self.start_stop_list_train, directory_list, non_nan_columns, lead)
            X_va, y_va = self.data_assembly(self.start_stop_list_val,   directory_list, non_nan_columns, lead)

            regr = sklearn.linear_model.LinearRegression().fit(X_tr, y_tr)
            y_hat = pd.Series(regr.predict(X_va), index=y_va.index).sort_index()
            y_va = y_va.sort_index()

            axs[i].plot(y_va, label='Validation', linewidth=2.2, color='blue')
            axs[i].plot(y_hat, label='Prediction', linewidth=2.2, color='red')

            axs[i].set_title(f'Surface Data Predicting Nino 3.4 Anomaly at {lead} Month Lead Time', fontsize=14)
            axs[i].set_ylabel('Nino 3.4 Anomaly', fontsize=14)

            axs[i].set_xlim(y_va.index[0], y_va.index[-1])
            axs[i].margins(x=0)

            axs[i].set_ylim(y_min, y_max)
            axs[i].yaxis.set_major_locator(mticker.MultipleLocator(1))

            axs[i].grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
            axs[i].tick_params(labelsize=11)

            axs[i].text(-0.06, 1.02, labels[i], transform=axs[i].transAxes,
                        ha='right', va='bottom', fontsize=12, fontweight='bold')

            axs[i].legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=10)

        axs[-1].set_xlabel('Year', fontsize=16)
        plt.savefig(os.path.join('figures', 'figure4_surface_predictions.png'), dpi=400, bbox_inches='tight')
        plt.show()

    def generate_figure5(self):
        import matplotlib.ticker as mticker

        predictor_depth = 115
        predictand_depth = 5
        lead_times = [1, 3, 6, 9]
        directory_list = self.obtain_dir(
            predictor_depth=predictor_depth, predictand_depth=predictand_depth
        )

        ds = xr.open_dataset(directory_list[0])
        deepTemp = ds['deepTemp'].values.reshape(ds['deepTemp'].shape[0], -1)
        non_nan_columns = ~np.isnan(deepTemp).any(axis=0)
        ds.close()

        fig, axs = plt.subplots(len(lead_times), 1, figsize=(12, 10), sharex=True, constrained_layout=True)
        labels = ["(a)", "(b)", "(c)", "(d)"]
        y_min, y_max = -2.5, 2.5

        for i, lead in enumerate(lead_times):
            X_tr, y_tr = self.data_assembly(self.start_stop_list_train, directory_list, non_nan_columns, lead)
            X_va, y_va = self.data_assembly(self.start_stop_list_val,   directory_list, non_nan_columns, lead)

            regr = sklearn.linear_model.LinearRegression().fit(X_tr, y_tr)
            y_hat = pd.Series(regr.predict(X_va), index=y_va.index).sort_index()
            y_va = y_va.sort_index()

            axs[i].plot(y_va, label='Validation', linewidth=2.2, color='blue')
            axs[i].plot(y_hat, label='Prediction', linewidth=2.2, color='red')

            axs[i].set_title(f'Deep Ocean Data Predicting Nino 3.4 Anomaly at {lead} Month Lead Time', fontsize=14)
            axs[i].set_ylabel('Nino 3.4 Anomaly', fontsize=14)

            axs[i].set_xlim(y_va.index[0], y_va.index[-1])
            axs[i].margins(x=0)

            axs[i].set_ylim(y_min, y_max)
            axs[i].yaxis.set_major_locator(mticker.MultipleLocator(1))

            axs[i].grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
            axs[i].tick_params(labelsize=11)

            axs[i].text(-0.06, 1.02, labels[i], transform=axs[i].transAxes,
                        ha='right', va='bottom', fontsize=12, fontweight='bold')

            axs[i].legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=10)

        axs[-1].set_xlabel('Year', fontsize=12)
        plt.savefig(os.path.join('figures', 'figure5_subsurface_predictions.png'), dpi=400, bbox_inches='tight')
        plt.show()

    def generate_figure6(self):
        import matplotlib.ticker as mticker

        predictor_depth = 115
        predictand_depth = 115
        lead_times = [1, 3, 6, 9]
        directory_list = self.obtain_dir(
            predictor_depth=predictor_depth, predictand_depth=predictand_depth
        )

        ds = xr.open_dataset(directory_list[0])
        deepTemp = ds['deepTemp'].values.reshape(ds['deepTemp'].shape[0], -1)
        non_nan_columns = ~np.isnan(deepTemp).any(axis=0)
        ds.close()

        with open(directory_list[1]) as f:
            vals = []
            for line in f:
                vals.extend(map(float, line.split()[1:]))
        y_series = pd.Series(vals, index=pd.date_range(self.dataset_begin, freq='MS', periods=len(vals)))

        fig, axs = plt.subplots(len(lead_times), 1, figsize=(12, 10), sharex=True, constrained_layout=True)
        labels = ["(a)", "(b)", "(c)", "(d)"]
        y_min, y_max = -5, 3

        for i, lead in enumerate(lead_times):
            X_tr, y_tr = self.data_assembly(self.start_stop_list_train, directory_list, non_nan_columns, lead)
            X_va, _ = self.data_assembly(self.start_stop_list_val, directory_list, non_nan_columns, lead)

            start_val = pd.to_datetime(self.start_stop_list_val[0]) + pd.DateOffset(months=lead)
            end_val = pd.to_datetime(self.start_stop_list_val[1]) + pd.DateOffset(months=lead)
            y_va = y_series[start_val:end_val][:len(X_va)]

            regr = sklearn.linear_model.LinearRegression().fit(X_tr, y_tr)
            y_hat = pd.Series(regr.predict(X_va), index=y_va.index).sort_index()
            y_va = y_va.sort_index()

            axs[i].plot(y_va, label='Validation', linewidth=2.2, color='blue')
            axs[i].plot(y_hat, label='Prediction', linewidth=2.2, color='red')

            axs[i].set_title(f'Deep Ocean Data Predicting Nino 3.4 Anomaly at {lead} Month Lead Time', fontsize=14)
            axs[i].set_ylabel('Deep Ocean Anomaly', fontsize=12)

            axs[i].set_xlim(y_va.index[0], y_va.index[-1])
            axs[i].margins(x=0)

            axs[i].set_ylim(y_min, y_max)
            axs[i].yaxis.set_major_locator(mticker.MultipleLocator(1))

            axs[i].grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
            axs[i].tick_params(labelsize=11)

            axs[i].text(-0.06, 1.02, labels[i], transform=axs[i].transAxes,
                        ha='right', va='bottom', fontsize=12, fontweight='bold')

            axs[i].legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=10)

        axs[-1].set_xlabel('Year', fontsize=12)
        plt.savefig(os.path.join('figures', 'figure6_deep_predictions.png'), dpi=400, bbox_inches='tight')
        plt.show()


# Heatmaps for Figures 7 & 8
def _flip_depth(arr_2d):
    """Flip depth axis so shallow (5 m) is at the top."""
    return np.flip(arr_2d.T, axis=0)


def corr_paper_cmap():
    """Your original diverging color scheme for correlation."""
    return LinearSegmentedColormap.from_list(
        "corr_paper",
        [
            (0.0, '#2c00ff'),   # deep blue
            (0.25, '#00f6ff'),  # cyan
            (0.5, '#ffffff'),   # white
            (0.75, '#fff800'),  # yellow
            (1.0, '#ff0000'),   # red
        ]
    )


def plot_correlation_heatmap(corr_mat, month_vec, depth_vec, fig_path,
                             title="Prediction skill: Correlation"): #Adjust titles as needed (can be customized just must be run multiple times)
    """Separated correlation heatmap (kept with your custom cmap)."""
    fig, ax = plt.subplots(figsize=(8.5, 7))
    data = _flip_depth(corr_mat)

    im = ax.imshow(data, aspect="auto", origin="upper",
                   cmap=corr_paper_cmap(), vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(month_vec)))
    ax.set_xticklabels(month_vec, fontsize=12)
    ax.set_yticks(np.arange(len(depth_vec)))
    ax.set_yticklabels(depth_vec, fontsize=12)
    ax.set_xlabel("Lead Time [months]")
    ax.set_ylabel("Depth [m]")
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax, shrink=0.95, pad=0.015)
    cbar.set_label("Correlation", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    ax.grid(False)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=400, bbox_inches="tight")
    plt.show()


def plot_rmse_heatmap(rmse_mat, month_vec, depth_vec, fig_path,
                      title="RMSE of Predictions"):  #Adjust titles as needed (can be customized just must be run multiple times)
    """Separated RMSE heatmap with monotone red palette."""
    fig, ax = plt.subplots(figsize=(8.5, 7))
    data = _flip_depth(rmse_mat)

    cmap_rmse = plt.get_cmap("Reds")
    vmin = 0.0
    vmax = float(np.nanmax(rmse_mat))

    im = ax.imshow(data, aspect="auto", cmap=cmap_rmse, vmin=vmin, vmax=vmax,
                   origin="upper")

    ax.set_xticks(np.arange(len(month_vec)))
    ax.set_xticklabels(month_vec, fontsize=12)
    ax.set_yticks(np.arange(len(depth_vec)))
    ax.set_yticklabels(depth_vec, fontsize=12)
    ax.set_xlabel("Lead Time [months]")
    ax.set_ylabel("Depth [m]")
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax, shrink=0.95, pad=0.015)
    cbar.set_label("RMSE (°C)", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    ax.grid(False)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=400, bbox_inches="tight")
    plt.show()


# Main (comment out 'generate_figure#' as needed)
if __name__ == "__main__":
    start = time.time()

    #generate_figure2_like_paper()  # Figure 2
    #generate_figure3_like_paper()  # Figure 3

    obj = EnsoLinearModel()

    #obj.generate_figure1_plotly_volume_like()  # Figure 1
    #obj.generate_figure4()  # Figure 4
    #obj.generate_figure5()  # Figure 5
    #obj.generate_figure6()  # Figure 6

    # Figure 7
    try:
        corr7 = np.load("stats/corr_fig7.npy")
        rmse7 = np.load("stats/rmse_fig7.npy")
        months7 = np.load("stats/months_fig7.npy")
        depths7 = np.load("stats/depths_fig7.npy")
    except Exception:
        corr7, rmse7, months7, depths7 = obj.run_all_scenarios(
            predictand_depth=5, use_deep_target=False
        )
        np.save("stats/corr_fig7.npy", corr7)
        np.save("stats/rmse_fig7.npy", rmse7)
        np.save("stats/months_fig7.npy", months7)
        np.save("stats/depths_fig7.npy", depths7)
        print("Computed and saved data for Figure 7.")

    plot_correlation_heatmap(corr7, months7, depths7, "figures/figure7_correlation.png") # Figure 7
    plot_rmse_heatmap(rmse7, months7, depths7, "figures/figure7_rmse.png") # Figure 7

    # Figure 8
    try:
        corr8 = np.load("stats/corr_fig8.npy")
        rmse8 = np.load("stats/rmse_fig8.npy")
        months8 = np.load("stats/months_fig8.npy")
        depths8 = np.load("stats/depths_fig8.npy")
    except Exception:
        corr8, rmse8, months8, depths8 = obj.run_all_scenarios(
            predictand_depth=115, use_deep_target=True
        )
        np.save("stats/corr_fig8.npy", corr8)
        np.save("stats/rmse_fig8.npy", rmse8)
        np.save("stats/months_fig8.npy", months8)
        np.save("stats/depths_fig8.npy", depths8)
        print("Computed and saved data for Figure 8.")

    plot_correlation_heatmap(corr8, months8, depths8, "figures/figure8_correlation.png") # Figure 8
    plot_rmse_heatmap(rmse8, months8, depths8, "figures/figure8_rmse.png") # Figure 8

    print(f'Finished in: {round(time.time() - start)} seconds')
