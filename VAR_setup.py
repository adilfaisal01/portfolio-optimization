import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import pandas as pd

# setting up training and test data
class VARAnalysis:
    def __init__(self, final_date,return_data=pd.read_csv(filepath_or_buffer="../Daily-return-ratio.csv")):
        self.train_final_date=final_date
        self.return_data=return_data
        self.indices_list=['AGG', 'GLD', 'SLV', 'SPY', 'VTI','VEA', 'VWO']
        self.parameters=None
        self.coeffs=None
        self.k_ar= None
        self.intercepts=None
    def data_segmentation(self):
        #training and test sets
        train_data=self.return_data[self.return_data['Date']<self.train_final_date]
        test_data=self.return_data[self.return_data['Date']>=self.train_final_date]
        return train_data, test_data
#check for stationarity, using the Augmented-DickeyFuller Test, more negative means more stattionary data, which is good
    def stationary_check(self, DS):
        for entry in self.indices_list:
            results=adfuller(DS[entry],autolag="AIC")
            print(f'Asset class: {entry}') 
            print(f'adf stats:{results[0]}')
            print(f'p-value:{results[1]}')
            # print(f'used_lags:{results[2]}')
            print(f'critical values:{results[4]}')
            print('\n')
    def fit_model(self,train_data):
        # vector autoregression
        TD=np.array(train_data[self.indices_list])
        model=VAR(TD)
        
        if self.k_ar is None:
            self.parameters=model.fit(ic='aic')
            self.k_ar=self.parameters.k_ar
            self.coeffs=self.parameters.coefs
            self.intercepts=self.parameters.intercept
        else:
            self.parameters=model.fit(ic='aic',maxlags=self.k_ar)
            self.k_ar=self.parameters.k_ar
            self.coeffs=self.parameters.coefs
            self.intercepts=self.parameters.intercept
    def forecast_model(self,N_horizon,lookbackdata):
        #forecasting using VAR
        predictions_return=self.parameters.forecast(y=lookbackdata,steps=N_horizon) #steps denote the number of days im forecasting
        pred_return_covariance=self.parameters.forecast_cov(steps=N_horizon) #covariance prediction
        return predictions_return, pred_return_covariance[0]
        
        
# # find the results of the VAR
# print(f'Order of lag: {parameters.k_ar}')
# print(f'Coefficients: {parameters.coefs}')
# print(f'Intercept values:{parameters.intercept}')
# print(f'shape of c: {parameters.intercept.shape}')
# print(f'shape of phi:{parameters.coefs[0].shape}')
# print(predictions_return.shape)
# print(pred_return_covariance.shape)

# xx=np.vstack((TD,predictions_return))
# print(xx.shape)

# var=VARAnalysis(final_date="2020-01-01")