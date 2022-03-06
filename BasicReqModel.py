import numpy as np
import logging
import tensorflow as tf
import matplotlib.pyplot as plt

logger = tf.get_logger()
logger.setLevel(logging.ERROR)


inp = [1 ,2 ,3 ,4 ,6 , 8, 20]
out = [99, 108, 117, 126, 144, 162, 270] 

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
l1 = tf.keras.layers.Dense(units=2)
l2 = tf.keras.layers.Dense(units=1)
model_1 = tf.keras.Sequential([l0, l1, l2])
model_1.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
history_1 = model_1.fit(inp, out, epochs=500, verbose=True)
print("Finished training the model")

print("These are the l0 variables: {}".format(l0.get_weights()))
print("These are the l1 variables: {}".format(l1.get_weights()))
print("These are the l2 variables: {}".format(l2.get_weights()))

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(hitory_2.history['loss'])

print(model_1.predict([6]))
print(model_1.predict([100]))
print(model_1.predict([124]))