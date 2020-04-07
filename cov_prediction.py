import numpy as np
import pandas as pd
import math
import datetime
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
#read data
data=pd.read_csv("C:/cov/covid-19/data/time-series-19-covid-combined.csv")
US_data=data.loc[data.loc[:,'Country/Region']=='US']
US_data.index=[i for i in range (US_data.shape[0])]

fig,ax = plt.subplots(1,1,figsize=(4,3));
ax.plot(US_data.index, US_data.Confirmed);

startTime = datetime.datetime.strptime('2020-01-22', "%Y-%m-%d")
endTime=datetime.datetime.strptime('2020-04-04', "%Y-%m-%d")
timerange=np.array([i for i in range((startTime - startTime).days,(endTime - startTime).days+1)])
class InfectProb:
    def __init__(self, timerange, nContact, gamma):
        self.timerange = timerange
        self.nContact, self.gamma = nContact, gamma
        self.data=US_data
    def obfunction(self,infectionProb):
        vector=np.array(np.exp((infectionProb*self.nContact-self.gamma)*self.timerange)-self.data.loc[self.timerange,'Confirmed'])
        return ((vector**2).sum())
    
ysx=InfectProb(timerange, 5 ,1/14)
result=minimize(ysx.obfunction, 0.04, method='nelder-mead', options={'xtol': 1e-8, 'disp': True}).x

class SIRModel:
    def __init__(self, N, beta, gamma):
        self.beta= beta
        self.gamma=gamma
        self.N=N
        self.t=np.linspace(0,360,361)
    def odemodel(self, population,t):
        diff=np.zeros(3)
        s,i,r=population
        diff[0] = - self.beta * s * i / self.N
        diff[1] = self.beta * s * i / self.N - self.gamma * i
        diff[2] = self.gamma * i
        return diff
beta, gamma, N = 4.5*result[0],1/14,327.2*100*10000
sa=SIRModel(N,beta,gamma)
od_result=odeint(sa.odemodel,[N-1,1,0],np.linspace(0,360,361))
cumulate_predict_confirm=[od_result[0:i,1].sum() for i in range(od_result.shape[0]+1)]
plt.plot(cumulate_predict_confirm[0:80],color = 'orange',label = 'Infection',marker = '.')
plt.plot(US_data.Confirmed,color = 'red',label = 'realdata',marker = '.')

plt.plot(od_result[:,2],color = 'green',label = 'recover',marker = '.')
plt.plot(US_data.Recovered,color = 'yellow',label = 'realdata',marker = '.')
plt.legend()

####
US_data_selected= US_data.loc[:,['Date','Confirmed','Recovered','Deaths']]
US_data_selected.plot()
#dfdata=US_data_selected.set_index("Date")
#dfdiff = dfdata.diff(periods=1).dropna()
#dfdiff = dfdiff.reset_index("Date")
#diff_data_merge=pd.merge(US_data_selected, dfdiff,on=['Date','Date'],how='outer')

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

