import numpy as np
import pandas as pd
import math
import datetime
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
data=pd.read_csv("C:/cov/covid-19/data/time-series-19-covid-combined.csv")
US_data=data.loc[data.loc[:,'Country/Region']=='US']
US_data.index=[i for i in range (US_data.shape[0])]
df=pd.DataFrame(US_data.Confirmed)
#data_ltsm=pd.read_csv("C:/Users/86953/Desktop/1.csv")
#data_ltsm.plot()
#np.random.seed(7)
#df=pd.DataFrame(data_ltsm.Passengers)

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(df)
#data=np.array(df)
#train_size = int(len(data) * 0.67)
#test_size = len(data) - train_size
#train, test = data[0:train_size,:], data[train_size:len(data),:]

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

## reshape into X=t and Y=t+1
#look_back = 2
#trainX, trainY = create_dataset(train, look_back)


# reshape input to be [samples, time steps, features]
#trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))
#testX = np.reshape(testX, (testX.shape[0], testX.shape[1],1))

# reshape input to be [samples, time steps, features]


# create and fit the LSTM network
#model = Sequential()
#model.add(LSTM(4, input_shape=(look_back,1)))
#model.add(Dense(1))
#model.compile(loss='mean_squared_error', optimizer='adam')
#model.fit(trainX, trainY, epochs=35, batch_size=1, verbose=2)

def ysx(dataset,look_back=1):
    dataX,dataY=[],[]
    loop=look_back-1
    for i in range(len(dataset)-loop):
        a=dataset[i:(i+look_back),0]
        dataX.append(a)
    return np.array(dataX)
#predictX= ysx(data, look_back)
#testPredict = model.predict(testX)
predict_result=[]
for i in range(10):
    look_back=i+1
    trainX, trainY = create_dataset(data, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))
    model = Sequential()
    #输入数据的shape为(n_samples, timestamps, features)
    #隐藏层设置为256, input_shape元组第二个参数1意指features为1
    #下面还有个lstm，故return_sequences设置为True
    model.add(LSTM(units=256,input_shape=(None,1),return_sequences=True))
    model.add(LSTM(units=256))
    #后接全连接层，直接输出单个值，故units为1
    model.add(Dense(units=1))
    #model.add(Activation('linear'))
    model.compile(loss='mse',optimizer='adam')
    model.fit(trainX, trainY, epochs=35, batch_size=1)
    predict_dataX= ysx(data, look_back)
    predict_dataX = np.reshape(predict_dataX, (predict_dataX.shape[0], predict_dataX.shape[1],1))
    preidct_scale=model.predict(predict_dataX)
    predict_not_scale=scaler.inverse_transform(preidct_scale)
    predict_result.append(predict_not_scale[-1])
#testPredict = scaler.inverse_transform(testPredict)
#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])
df_predict=pd.DataFrame(predict_result,columns=['Confirmed'], index=[i+df.shape[0] for i in range(len(predict_result))])    

plt.plot(df,color = 'orange',label = 'Infection',marker = '.')
plt.plot(df_predict,color = 'orange',label = 'Infection',marker = '.')
plt.show()
plt.plot(trainY.reshape(-1,1),color = 'red',label = 'realdata',marker = '.')
plt.plot(testY.reshape(-1,1),color = 'red',label = 'realdata',marker = '.')
plt.plot(testPredict.reshape(-1,1),color = 'green',label = 'realdata',marker = '.')
data1,y=create_dataset(data, look_back)
data1 = np.reshape(data1, (data1.shape[0], data1.shape[1],1))
ysx=model.predict(data1)
ysx=scaler.inverse_transform(ysx)
