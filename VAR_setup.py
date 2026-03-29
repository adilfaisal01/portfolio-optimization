import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import pandas as pd

# setting up training and test data
return_data=pd.read_csv(filepath_or_buffer="../Daily-return-ratio.csv")
train_final_date="2020-01-01"

#training and test sets
train_data=return_data[return_data['Date']<train_final_date]
test_data=return_data[return_data['Date']>=train_final_date]

indices_list=['AGG', 'GLD', 'SLV', 'SPY', 'VTI','VEA', 'VWO']

#check for stationarity, using the Augmented-DickeyFuller Test, more negative means more stattionary data, which is good
def stationary_check(DS, indices_list):
    for entry in indices_list:
        results=adfuller(DS[entry],autolag="AIC")
        print(f'Asset class: {entry}') 
        print(f'adf stats:{results[0]}')
        print(f'p-value:{results[1]}')
        # print(f'used_lags:{results[2]}')
        print(f'critical values:{results[4]}')
        print('\n')


stationarity_criteria=stationary_check(train_data,indices_list)

# Vector autoregression fitting
TD=np.array(train_data[indices_list])
model=VAR(TD)
parameters=model.fit(ic='aic') 

# find the results of the VAR
print(f'Order of lag: {parameters.k_ar}')
print(f'Coefficients: {parameters.coefs}')
print(f'Intercept values:{parameters.intercept}')



print(f'shape of c: {parameters.intercept.shape}')
print(f'shape of phi:{parameters.coefs[0].shape}')

#forecasting using VAR
predictions_return=parameters.forecast(TD[-parameters.k_ar:],steps=100) #steps denote the number of days im forecasting
pred_return_covariance=parameters.forecast_cov(steps=1) #covariance prediction 

# xx=np.vstack((TD,predictions_return))
# print(xx.shape)









