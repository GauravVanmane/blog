
# Title:Predicting appearance of sunspots using LSTMs.


```
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
```


```
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
```


```
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)  
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000
```


```
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset
```


```
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

train_set = windowed_dataset(x_train, window_size, batch_size=128, shuffle_buffer=shuffle_buffer_size)

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1)),
        tf.keras.layers.SimpleRNN(40, return_sequences = True),
        tf.keras.layers.SimpleRNN(40),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x : x *100.0)

])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr = 1e-8, momentum = 0.9)

model.compile(loss = tf.keras.losses.Huber(),
              optimizer = optimizer,
              metrics = ['mae'])

summary = model.fit(train_set, epochs = 100, callbacks = [lr_schedule])

```

    Epoch 1/100
    8/8 [==============================] - 0s 50ms/step - loss: 195.5726 - mae: 196.0726 - lr: 1.0000e-08
    Epoch 2/100
    8/8 [==============================] - 0s 43ms/step - loss: 194.7820 - mae: 195.2820 - lr: 1.1220e-08
    Epoch 3/100
    8/8 [==============================] - 0s 44ms/step - loss: 193.5110 - mae: 194.0110 - lr: 1.2589e-08
    Epoch 4/100
    8/8 [==============================] - 0s 44ms/step - loss: 191.9081 - mae: 192.4081 - lr: 1.4125e-08
    Epoch 5/100
    8/8 [==============================] - 0s 43ms/step - loss: 190.0235 - mae: 190.5235 - lr: 1.5849e-08
    Epoch 6/100
    8/8 [==============================] - 0s 42ms/step - loss: 187.8583 - mae: 188.3583 - lr: 1.7783e-08
    Epoch 7/100
    8/8 [==============================] - 0s 43ms/step - loss: 185.3787 - mae: 185.8787 - lr: 1.9953e-08
    Epoch 8/100
    8/8 [==============================] - 0s 43ms/step - loss: 182.5484 - mae: 183.0484 - lr: 2.2387e-08
    Epoch 9/100
    8/8 [==============================] - 0s 44ms/step - loss: 179.3127 - mae: 179.8127 - lr: 2.5119e-08
    Epoch 10/100
    8/8 [==============================] - 0s 46ms/step - loss: 175.6017 - mae: 176.1017 - lr: 2.8184e-08
    Epoch 11/100
    8/8 [==============================] - 0s 45ms/step - loss: 171.3244 - mae: 171.8244 - lr: 3.1623e-08
    Epoch 12/100
    8/8 [==============================] - 0s 46ms/step - loss: 166.3522 - mae: 166.8522 - lr: 3.5481e-08
    Epoch 13/100
    8/8 [==============================] - 0s 43ms/step - loss: 160.5518 - mae: 161.0518 - lr: 3.9811e-08
    Epoch 14/100
    8/8 [==============================] - 0s 45ms/step - loss: 153.6379 - mae: 154.1379 - lr: 4.4668e-08
    Epoch 15/100
    8/8 [==============================] - 0s 46ms/step - loss: 145.3262 - mae: 145.8262 - lr: 5.0119e-08
    Epoch 16/100
    8/8 [==============================] - 0s 44ms/step - loss: 135.0352 - mae: 135.5352 - lr: 5.6234e-08
    Epoch 17/100
    8/8 [==============================] - 0s 41ms/step - loss: 121.9921 - mae: 122.4921 - lr: 6.3096e-08
    Epoch 18/100
    8/8 [==============================] - 0s 46ms/step - loss: 105.3427 - mae: 105.8427 - lr: 7.0795e-08
    Epoch 19/100
    8/8 [==============================] - 0s 43ms/step - loss: 84.1256 - mae: 84.6256 - lr: 7.9433e-08
    Epoch 20/100
    8/8 [==============================] - 0s 44ms/step - loss: 57.2713 - mae: 57.7713 - lr: 8.9125e-08
    Epoch 21/100
    8/8 [==============================] - 0s 43ms/step - loss: 32.0324 - mae: 32.5290 - lr: 1.0000e-07
    Epoch 22/100
    8/8 [==============================] - 0s 47ms/step - loss: 24.3013 - mae: 24.7973 - lr: 1.1220e-07
    Epoch 23/100
    8/8 [==============================] - 0s 41ms/step - loss: 23.3490 - mae: 23.8462 - lr: 1.2589e-07
    Epoch 24/100
    8/8 [==============================] - 0s 44ms/step - loss: 21.1136 - mae: 21.6102 - lr: 1.4125e-07
    Epoch 25/100
    8/8 [==============================] - 0s 46ms/step - loss: 17.1272 - mae: 17.6220 - lr: 1.5849e-07
    Epoch 26/100
    8/8 [==============================] - 0s 44ms/step - loss: 13.8473 - mae: 14.3365 - lr: 1.7783e-07
    Epoch 27/100
    8/8 [==============================] - 0s 47ms/step - loss: 10.7750 - mae: 11.2631 - lr: 1.9953e-07
    Epoch 28/100
    8/8 [==============================] - 0s 44ms/step - loss: 8.7659 - mae: 9.2524 - lr: 2.2387e-07
    Epoch 29/100
    8/8 [==============================] - 0s 42ms/step - loss: 7.9784 - mae: 8.4596 - lr: 2.5119e-07
    Epoch 30/100
    8/8 [==============================] - 0s 46ms/step - loss: 7.8824 - mae: 8.3651 - lr: 2.8184e-07
    Epoch 31/100
    8/8 [==============================] - 0s 42ms/step - loss: 7.8424 - mae: 8.3231 - lr: 3.1623e-07
    Epoch 32/100
    8/8 [==============================] - 0s 43ms/step - loss: 7.7293 - mae: 8.2061 - lr: 3.5481e-07
    Epoch 33/100
    8/8 [==============================] - 0s 47ms/step - loss: 7.6896 - mae: 8.1682 - lr: 3.9811e-07
    Epoch 34/100
    8/8 [==============================] - 0s 48ms/step - loss: 7.5853 - mae: 8.0646 - lr: 4.4668e-07
    Epoch 35/100
    8/8 [==============================] - 0s 42ms/step - loss: 7.5172 - mae: 7.9977 - lr: 5.0119e-07
    Epoch 36/100
    8/8 [==============================] - 0s 43ms/step - loss: 7.4070 - mae: 7.8849 - lr: 5.6234e-07
    Epoch 37/100
    8/8 [==============================] - 0s 46ms/step - loss: 7.3480 - mae: 7.8265 - lr: 6.3096e-07
    Epoch 38/100
    8/8 [==============================] - 0s 41ms/step - loss: 7.2938 - mae: 7.7685 - lr: 7.0795e-07
    Epoch 39/100
    8/8 [==============================] - 0s 44ms/step - loss: 7.2491 - mae: 7.7239 - lr: 7.9433e-07
    Epoch 40/100
    8/8 [==============================] - 0s 42ms/step - loss: 7.2260 - mae: 7.7028 - lr: 8.9125e-07
    Epoch 41/100
    8/8 [==============================] - 0s 46ms/step - loss: 7.2844 - mae: 7.7629 - lr: 1.0000e-06
    Epoch 42/100
    8/8 [==============================] - 0s 42ms/step - loss: 7.4529 - mae: 7.9379 - lr: 1.1220e-06
    Epoch 43/100
    8/8 [==============================] - 0s 46ms/step - loss: 7.1585 - mae: 7.6390 - lr: 1.2589e-06
    Epoch 44/100
    8/8 [==============================] - 0s 44ms/step - loss: 7.1436 - mae: 7.6195 - lr: 1.4125e-06
    Epoch 45/100
    8/8 [==============================] - 0s 45ms/step - loss: 7.0569 - mae: 7.5397 - lr: 1.5849e-06
    Epoch 46/100
    8/8 [==============================] - 0s 44ms/step - loss: 6.9329 - mae: 7.4142 - lr: 1.7783e-06
    Epoch 47/100
    8/8 [==============================] - 0s 43ms/step - loss: 6.7453 - mae: 7.2217 - lr: 1.9953e-06
    Epoch 48/100
    8/8 [==============================] - 0s 42ms/step - loss: 7.4523 - mae: 7.9375 - lr: 2.2387e-06
    Epoch 49/100
    8/8 [==============================] - 0s 41ms/step - loss: 7.2814 - mae: 7.7655 - lr: 2.5119e-06
    Epoch 50/100
    8/8 [==============================] - 0s 44ms/step - loss: 6.9146 - mae: 7.3947 - lr: 2.8184e-06
    Epoch 51/100
    8/8 [==============================] - 0s 45ms/step - loss: 7.0549 - mae: 7.5381 - lr: 3.1623e-06
    Epoch 52/100
    8/8 [==============================] - 0s 40ms/step - loss: 7.4218 - mae: 7.9044 - lr: 3.5481e-06
    Epoch 53/100
    8/8 [==============================] - 0s 40ms/step - loss: 6.9011 - mae: 7.3845 - lr: 3.9811e-06
    Epoch 54/100
    8/8 [==============================] - 0s 46ms/step - loss: 6.7601 - mae: 7.2387 - lr: 4.4668e-06
    Epoch 55/100
    8/8 [==============================] - 0s 46ms/step - loss: 6.6574 - mae: 7.1361 - lr: 5.0119e-06
    Epoch 56/100
    8/8 [==============================] - 0s 44ms/step - loss: 6.1865 - mae: 6.6627 - lr: 5.6234e-06
    Epoch 57/100
    8/8 [==============================] - 0s 45ms/step - loss: 6.8587 - mae: 7.3385 - lr: 6.3096e-06
    Epoch 58/100
    8/8 [==============================] - 0s 45ms/step - loss: 7.6405 - mae: 8.1257 - lr: 7.0795e-06
    Epoch 59/100
    8/8 [==============================] - 0s 45ms/step - loss: 7.4379 - mae: 7.9262 - lr: 7.9433e-06
    Epoch 60/100
    8/8 [==============================] - 0s 44ms/step - loss: 7.3505 - mae: 7.8368 - lr: 8.9125e-06
    Epoch 61/100
    8/8 [==============================] - 0s 42ms/step - loss: 7.3676 - mae: 7.8506 - lr: 1.0000e-05
    Epoch 62/100
    8/8 [==============================] - 0s 39ms/step - loss: 11.3233 - mae: 11.8168 - lr: 1.1220e-05
    Epoch 63/100
    8/8 [==============================] - 0s 42ms/step - loss: 10.1967 - mae: 10.6869 - lr: 1.2589e-05
    Epoch 64/100
    8/8 [==============================] - 0s 41ms/step - loss: 7.2608 - mae: 7.7468 - lr: 1.4125e-05
    Epoch 65/100
    8/8 [==============================] - 0s 40ms/step - loss: 10.9818 - mae: 11.4748 - lr: 1.5849e-05
    Epoch 66/100
    8/8 [==============================] - 0s 46ms/step - loss: 11.1194 - mae: 11.6105 - lr: 1.7783e-05
    Epoch 67/100
    8/8 [==============================] - 0s 43ms/step - loss: 10.5319 - mae: 11.0232 - lr: 1.9953e-05
    Epoch 68/100
    8/8 [==============================] - 0s 40ms/step - loss: 11.4124 - mae: 11.9049 - lr: 2.2387e-05
    Epoch 69/100
    8/8 [==============================] - 0s 43ms/step - loss: 10.6386 - mae: 11.1294 - lr: 2.5119e-05
    Epoch 70/100
    8/8 [==============================] - 0s 43ms/step - loss: 8.3821 - mae: 8.8703 - lr: 2.8184e-05
    Epoch 71/100
    8/8 [==============================] - 0s 47ms/step - loss: 7.7766 - mae: 8.2639 - lr: 3.1623e-05
    Epoch 72/100
    8/8 [==============================] - 0s 43ms/step - loss: 6.6853 - mae: 7.1687 - lr: 3.5481e-05
    Epoch 73/100
    8/8 [==============================] - 0s 43ms/step - loss: 7.7178 - mae: 8.2052 - lr: 3.9811e-05
    Epoch 74/100
    8/8 [==============================] - 0s 43ms/step - loss: 7.6203 - mae: 8.1101 - lr: 4.4668e-05
    Epoch 75/100
    8/8 [==============================] - 0s 40ms/step - loss: 12.5437 - mae: 13.0376 - lr: 5.0119e-05
    Epoch 76/100
    8/8 [==============================] - 0s 47ms/step - loss: 17.2193 - mae: 17.7140 - lr: 5.6234e-05
    Epoch 77/100
    8/8 [==============================] - 0s 45ms/step - loss: 13.2713 - mae: 13.7637 - lr: 6.3096e-05
    Epoch 78/100
    8/8 [==============================] - 0s 43ms/step - loss: 11.2212 - mae: 11.7101 - lr: 7.0795e-05
    Epoch 79/100
    8/8 [==============================] - 0s 43ms/step - loss: 10.2482 - mae: 10.7398 - lr: 7.9433e-05
    Epoch 80/100
    8/8 [==============================] - 0s 44ms/step - loss: 9.1872 - mae: 9.6765 - lr: 8.9125e-05
    Epoch 81/100
    8/8 [==============================] - 0s 44ms/step - loss: 10.7029 - mae: 11.1905 - lr: 1.0000e-04
    Epoch 82/100
    8/8 [==============================] - 0s 46ms/step - loss: 15.3374 - mae: 15.8323 - lr: 1.1220e-04
    Epoch 83/100
    8/8 [==============================] - 0s 45ms/step - loss: 22.2632 - mae: 22.7593 - lr: 1.2589e-04
    Epoch 84/100
    8/8 [==============================] - 0s 46ms/step - loss: 23.5502 - mae: 24.0481 - lr: 1.4125e-04
    Epoch 85/100
    8/8 [==============================] - 0s 43ms/step - loss: 25.6734 - mae: 26.1705 - lr: 1.5849e-04
    Epoch 86/100
    8/8 [==============================] - 0s 46ms/step - loss: 26.1727 - mae: 26.6714 - lr: 1.7783e-04
    Epoch 87/100
    8/8 [==============================] - 0s 47ms/step - loss: 14.7883 - mae: 15.2833 - lr: 1.9953e-04
    Epoch 88/100
    8/8 [==============================] - 0s 47ms/step - loss: 19.2865 - mae: 19.7825 - lr: 2.2387e-04
    Epoch 89/100
    8/8 [==============================] - 0s 42ms/step - loss: 38.1402 - mae: 38.6394 - lr: 2.5119e-04
    Epoch 90/100
    8/8 [==============================] - 0s 50ms/step - loss: 30.8946 - mae: 31.3926 - lr: 2.8184e-04
    Epoch 91/100
    8/8 [==============================] - 0s 41ms/step - loss: 34.9831 - mae: 35.4815 - lr: 3.1623e-04
    Epoch 92/100
    8/8 [==============================] - 0s 45ms/step - loss: 36.9047 - mae: 37.4036 - lr: 3.5481e-04
    Epoch 93/100
    8/8 [==============================] - 0s 46ms/step - loss: 45.9577 - mae: 46.4556 - lr: 3.9811e-04
    Epoch 94/100
    8/8 [==============================] - 0s 46ms/step - loss: 63.9257 - mae: 64.4256 - lr: 4.4668e-04
    Epoch 95/100
    8/8 [==============================] - 0s 42ms/step - loss: 22.4580 - mae: 22.9550 - lr: 5.0119e-04
    Epoch 96/100
    8/8 [==============================] - 0s 44ms/step - loss: 94.0082 - mae: 94.5070 - lr: 5.6234e-04
    Epoch 97/100
    8/8 [==============================] - 0s 46ms/step - loss: 37.5791 - mae: 38.0786 - lr: 6.3096e-04
    Epoch 98/100
    8/8 [==============================] - 0s 44ms/step - loss: 106.9641 - mae: 107.4627 - lr: 7.0795e-04
    Epoch 99/100
    8/8 [==============================] - 0s 44ms/step - loss: 296.5594 - mae: 297.0592 - lr: 7.9433e-04
    Epoch 100/100
    8/8 [==============================] - 0s 44ms/step - loss: 325.6585 - mae: 326.1585 - lr: 8.9125e-04
    


```
plt.semilogx(summary.history['lr'], summary.history['loss'])
plt.axis([1e-8,1e-4,0,30])
```




    (1e-08, 0.0001, 0.0, 30.0)




![png](img/Time_Series_Prediction_Week_3_Simple_RNN_Notebook_5_1.png)



```
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

dataset = windowed_dataset(x_train, window_size, batch_size=128, shuffle_buffer=shuffle_buffer_size)

model = tf.keras.models.Sequential([
            tf.keras.layers.Lambda(lambda x : tf.expand_dims(x, axis = -1)),
            tf.keras.layers.SimpleRNN(40, return_sequences = True),
            tf.keras.layers.SimpleRNN(40),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda( lambda x: x*100.0)

])

optimizer = tf.keras.optimizers.SGD(lr = 5e-5, momentum = 0.9)
model.compile(loss = tf.keras.losses.Huber(), optimizer = optimizer,
              metrics = ['mae'])
history = model.fit(dataset, epochs = 400)

```

    Epoch 1/400
    8/8 [==============================] - 0s 46ms/step - loss: 81.7535 - mae: 82.2535
    Epoch 2/400
    8/8 [==============================] - 0s 45ms/step - loss: 21.9357 - mae: 22.4301
    Epoch 3/400
    8/8 [==============================] - 0s 45ms/step - loss: 15.9200 - mae: 16.4144
    Epoch 4/400
    
         . .
         . .
         . .
         . .

    Epoch 396/400
    8/8 [==============================] - 0s 43ms/step - loss: 5.1800 - mae: 5.6585
    Epoch 397/400
    8/8 [==============================] - 0s 41ms/step - loss: 6.0542 - mae: 6.5386
    Epoch 398/400
    8/8 [==============================] - 0s 43ms/step - loss: 5.0056 - mae: 5.4852
    Epoch 399/400
    8/8 [==============================] - 0s 45ms/step - loss: 4.4122 - mae: 4.8904
    Epoch 400/400
    8/8 [==============================] - 0s 47ms/step - loss: 5.3008 - mae: 5.7832
    


```
forecast = []

for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time - window_size:]
results = np.array(forecast)[:,0,0]

plt.figure(figsize = (10,6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results)
```


![png](img/Time_Series_Prediction_Week_3_Simple_RNN_Notebook_7_0.png)



```
tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
```




    6.899144




```
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
mae=history.history['mae']
loss=history.history['loss']

epochs=range(len(loss)) # Get number of epochs

#------------------------------------------------
# Plot MAE and Loss
#------------------------------------------------
plt.plot(epochs, mae, 'r')
plt.plot(epochs, loss, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])

plt.figure()

epochs_zoom = epochs[200:]
mae_zoom = mae[200:]
loss_zoom = loss[200:]

#------------------------------------------------
# Plot Zoomed MAE and Loss
#------------------------------------------------
plt.plot(epochs_zoom, mae_zoom, 'r')
plt.plot(epochs_zoom, loss_zoom, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])

plt.figure()
```




    <Figure size 432x288 with 0 Axes>




![png](img/Time_Series_Prediction_Week_3_Simple_RNN_Notebook_9_1.png)



![png](img/Time_Series_Prediction_Week_3_Simple_RNN_Notebook_9_2.png)



    <Figure size 432x288 with 0 Axes>



```

```
