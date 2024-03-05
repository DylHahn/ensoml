from matplotlib import pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NHITS, Autoformer, Informer
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import scipy.stats
import sklearn.metrics, sklearn.linear_model
import time
import xarray as xr


class EnsoModel():
    def __init__(self):
        self.dataset_begin = '1980-01-01'
        self.dataset_end = '2017-12-31'
        self.start_stop_list_train = ['1980-01-01', '1995-12-31']   # Default train / validation date ranges
        self.start_stop_list_val = ['1997-01-01', '2006-12-31']     # Default train / validation date ranges
        self.save_time_series_plots = False
        for x in ['stats', 'figures']:
            if not(os.path.isdir(x)):
                os.mkdir(x)
        return
    
    def data_assembly(self, date_list, path_list, nan_mask, lead_time):
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
        figName_png = f'predictor_{predictor_depth}m_predictand_{predictand_depth}m_{lead}month_Predictions.png'
        figName_pdf = f'predictor_{predictor_depth}m_predictand_{predictand_depth}m_{lead}month_Predictions.pdf'
        predictions = pd.Series(predictions, index=y_true.index)
        predictions = predictions.sort_index()
        y_true = y_true.sort_index()
        plt.plot(y_true, label='Ground Truth')
        plt.plot(predictions, '--', label='LR Predictions')
        plt.legend(loc='best')
        plt.title(title)
        plt.ylabel(f'{predictand_depth}m Temperature Anomalies')
        plt.xlabel('Time (Months)')
        for f in [figName_png, figName_pdf]:
            plt.savefig(os.path.join('figures', f))
        plt.close()
        return

    def obtain_dir(self, predictor_depth=5, predictand_depth=5):
        return [os.path.join("data", f"godasClimatologyData_{predictor_depth}m.nc"), \
            os.path.join("data", f"movingAverageAnomalies{predictand_depth}m.txt")]

    def corr_rmse_fcn(self, y_true, predictions):
        return scipy.stats.pearsonr(y_true, predictions)[0], sklearn.metrics.mean_squared_error(y_true ,predictions)


class EnsoLinearModel(EnsoModel):
    def __init__(self):
        super(EnsoLinearModel, self).__init__()
        return

    def run_all_scenarios(self):
        month_vec = np.arange(1, 15)
        depth_vec = np.arange(5, 205, 10)
        corr_mat = np.zeros((len(month_vec), len(depth_vec)))
        rmse_mat = np.zeros((len(month_vec), len(depth_vec)))
        for mm in range(len(depth_vec)):
            depth_sel = depth_vec[mm]
            directory_list = self.obtain_dir(predictor_depth=depth_sel)
            ds = xr.open_dataset(directory_list[0])
            deepTemp = ds['deepTemp']
            deepTemp_reshape = deepTemp.values.reshape(deepTemp.shape[0], -1)
            non_nan_columns = ~np.isnan(deepTemp_reshape).any(axis=0)
            ds.close()
            for nn in range(len(month_vec)):
                X_train, y_train = self.data_assembly(self.start_stop_list_train, directory_list, non_nan_columns, month_vec[nn])
                X_val, y_val = self.data_assembly(self.start_stop_list_val, directory_list, non_nan_columns, month_vec[nn])
                regr = sklearn.linear_model.LinearRegression()
                regr.fit(X_train, y_train)
                y_predicted = regr.predict(X_val)
                corr, rmse = self.corr_rmse_fcn(y_val, y_predicted)
                corr_mat[nn, mm] = corr
                rmse_mat[nn, mm] = rmse
                if self.save_time_series_plots:
                    fig_title1 = f'Predicted and True Temperature Anomalies at {month_vec[nn]} MonthLead Time \n'
                    fig_title2 = 'Corr : %.2f, RMSE : %.2f %(corr, rmse)'
                    fig_title = fig_title1 + fig_title2
                    self.save_figs(y_val, y_predicted, depth_sel, month_vec[nn], fig_title)

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
        ds.close ()
        for idx in range(len(month_vec)):
            X_train, y_train = self.data_assembly(self.start_stop_list_train, directory_list, non_nan_columns, month_vec[idx])
            X_val, y_val = self.data_assembly(self.start_stop_list_val, directory_list, non_nan_columns, month_vec[idx])
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
    

class EnsoNeuralModel(EnsoModel):
    # Tools used
    # https://github.com/Nixtla/neuralforecast
    # https://nixtlaverse.nixtla.io/neuralforecast/examples/getting_started.html
    # https://colab.research.google.com/drive/1WjBbQzaivQhOldGolzymOtLmo6QX4Ieg#scrollTo=GUte-L1ZQQvr
    # Theory:
    # https://huggingface.co/blog/time-series-transformers
    # https://huggingface.co/blog/autoformer
    def __init__(self):
        super(EnsoNeuralModel, self).__init__()
        self.surface_anom_path = os.path.join('data', 'movingAverageAnomalies5m.txt')
        return
    
    def data_assembly(self, anomaly_path):
        with open(anomaly_path) as f:
            line = f.readline()
            enso_vals = []
            while line:
                yearly_enso_vals = map(float, line.split()[1:])
                enso_vals.extend(yearly_enso_vals)
                line = f.readline()
        return pd.DataFrame({
            'unique_id': [1 for _ in range(len(enso_vals))],
            'ds': pd.to_datetime(pd.date_range(self.dataset_begin, freq='MS', periods=len(enso_vals))),
            'y': pd.Series(enso_vals),
            })
    
    def run_nf_scenario(self):
        # TODO: What experiment is useful here? Lot of free params
        # Ideas:
            # Short term vs long term strength as a function of model
            # Can GODAS data be passed to models?
            # How sensitive is model performance to training region? e.g., should we train in regions w anomalies
        df = self.data_assembly(self.surface_anom_path)
        lb = pd.Timestamp(self.start_stop_list_train[0])
        ub = pd.Timestamp(self.start_stop_list_train[1])
        df_train = df.loc[(lb <= df['ds']) & (df['ds'] <= ub)]
        # horizon = 12
        horizon = 30
        # horizon = 36
        models = [
            LSTM(
                h=horizon,                    # Forecast horizon
                max_steps=500,                # Number of steps to train
                scaler_type='standard',       # Type of scaler to normalize data
                encoder_hidden_size=64,       # Defines the size of the hidden state of the LSTM
                decoder_hidden_size=64,       # Defines the number of hidden units of each layer of the MLP decoder
                ),
            NHITS(
                h=horizon,                   # Forecast horizon
                input_size=2 * horizon,      # Length of input sequence
                max_steps=500,               # Number of steps to train
                n_freq_downsample=[2, 1, 1], # Downsampling factors for each stack output
                ),
            # Informer(
            #     h=horizon,
            #     input_size=len(df_train) // 2,    # input_size=-1 is bugged?
            #     max_steps=100,
            #     )
            ]
        nf = NeuralForecast(models=models, freq='M')
        nf.fit(df=df_train)
        Y_hat_df = nf.predict()
        Y_hat_df = Y_hat_df.reset_index()
        df = pd.concat([
            pd.DataFrame({'ds': df['ds'], 'y': df['y'], 'Source': ['Ground Truth' for _ in range(len(df))]}),
            pd.DataFrame({'ds': df_train['ds'], 'y': df_train['y'], 'Source': ['Training Data' for _ in range(len(df_train['y']))]}),
            pd.DataFrame({'ds': Y_hat_df['ds'], 'y': Y_hat_df['LSTM'], 'Source': ['LSTM' for _ in range(len(Y_hat_df['ds']))]}),
            pd.DataFrame({'ds': Y_hat_df['ds'], 'y': Y_hat_df['NHITS'], 'Source': ['NHITS' for _ in range(len(Y_hat_df['ds']))]}),
            # pd.DataFrame({'ds': Y_hat_df['ds'], 'y': Y_hat_df['Informer'], 'Source': ['Informer' for _ in range(len(Y_hat_df['ds']))]}),
            ])
        df.to_csv(os.path.join('stats', 'nf_results.csv'), index=None)
        return
    
    def show_nf_plot(self):
        df_gt = self.data_assembly(self.surface_anom_path)
        df_nf = pd.read_csv(os.path.join('stats', 'nf_results.csv'))
        df_train = df_nf.loc[df_nf['Source'] == 'Training Data']
        df_lstm = df_nf.loc[df_nf['Source'] == 'LSTM']
        df_nhits = df_nf.loc[df_nf['Source'] == 'NHITS']
        fig = go.Figure()
        name_to_df = {'Ground Truth': df_gt, 'Training Data': df_train, 'LSTM': df_lstm, 'NHITS': df_nhits}
        for k, df in name_to_df.items():
            print(k, df.columns)
            fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name=k))
        fig.show()
        return

if __name__ == "__main__":
    start = time.time()

    obj = EnsoLinearModel()
    obj.run_single_scenario()

    obj = EnsoNeuralModel()
    obj.run_nf_scenario()
    obj.show_nf_plot()

    print(f'Finished in: {round(time.time() - start)} seconds')