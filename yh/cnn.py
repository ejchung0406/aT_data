import tensorflow as tf
import numpy as np

pummok = 1
x_train = np.loadtxt(f'../data/train_cnn/data_tr_{pummok}.csv', delimiter=",")
y_train = np.loadtxt(f'../data/train_cnn/label_tr_{pummok}.csv', delimiter=",")
x_train = x_train.reshape(-1, 14, 39, 1)
scale_bias = np.loadtxt('../data/train_cnn/scaler_bias.csv', delimiter=",")
scale_mult = np.loadtxt('../data/train_cnn/scaler_mult.csv', delimiter=",")

input_shape = (14, 39, 1)
batch_size = 8
epochs = 5

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (1, 39), activation='relu', input_shape=input_shape),
    tf.keras.layers.Reshape((14, 16, 1)),
    tf.keras.layers.Conv2D(32, (5, 5), padding='same',  activation='relu'),
    tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(1024, activation='relu'),
    # tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(28)
])

model.compile(optimizer='adam', loss='mean_absolute_error')
print(model.summary())

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'./cnn/cnn_train_{pummok}.ckpt',
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=[cp_callback],
                    shuffle=True)

tmp = model(x_train[0].reshape(1,14,39,1))
print(tmp * scale_mult[pummok][3] + scale_bias[pummok][3])
print(y_train[0] * scale_mult[pummok][3] + scale_bias[pummok][3])

# loss, acc = model.evaluate(x_test, y_test)
# print("Test accuracy: {:5.2f}%".format(100*acc))
