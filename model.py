from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os

class Transformer(keras.Model):
    def __init__(self, x_train, learning_rate):
        super().__init__()
        self.model = build_model(
        x_train.shape[1:],
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
        )

        self.model.compile(
            loss="mean_squared_error",
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        )
    
    def load_model(self, df_number, epoch, batch):
        if os.path.exists(f'./model') == False:
            os.mkdir(f'./model')
            
        model_path = f'./check/transformer-{df_number}-{epoch}-{batch}.h5'
        if os.path.exists(model_path) == True:
            self.model.load_weights(model_path)
            print(f"successfully loaded model {model_path}")

    def save_model(self, df_number, epoch, batch):
        # 모델 저장
        self.model.save(f'./model/transformer-{df_number}-{epoch}-{batch}.h5')

## Transformer 정의
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):

    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(28)(x) # 4주 예측
    return keras.Model(inputs, outputs)

## keras eraly stop, chekpoint 정의
def call_back_set(name, epoch, batch_size):
    early_stopping = EarlyStopping(monitor='val_loss', patience=100)

    if os.path.exists(f'./check') == False:
        os.mkdir(f'./check')

    filename = f'./check/{name}-{epoch}-{batch_size}.h5'

    checkpoint = ModelCheckpoint(filename,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='auto'
                                 )
    return [early_stopping, checkpoint]
