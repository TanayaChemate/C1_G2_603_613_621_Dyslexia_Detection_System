import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# --- 1. Global Parameters ---
BASE_DIR = 'data'
img_height, img_width = 128, 128
input_channels = 3
batch_size = 4   # small dataset → small batch
epochs_max = 20
model_name = 'model_lstm.h5'

# --- 2. Custom Data Generator for LSTM ---
def lstm_data_generator(generator):
    """Convert (batch,128,128,3) → (batch,128,384) for LSTM."""
    for X_batch, Y_batch in generator:
        batch_size_actual = X_batch.shape[0]
        sequence_length = img_height
        features_per_step = img_width * input_channels

        X_batch_reshaped = X_batch.reshape(
            batch_size_actual,
            sequence_length,
            features_per_step
        )

        yield X_batch_reshaped, Y_batch

# --- 3. LSTM Model ---
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.3))

    model.add(LSTM(64, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model

# --- 4. Data Loading ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'train'),
    target_size=(img_height, img_width),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    seed=42
)

validation_generator = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'train'),
    target_size=(img_height, img_width),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    seed=42
)

# --- 5. Build Model ---
input_shape = (img_height, img_width * input_channels)
model_lstm = create_lstm_model(input_shape)

model_lstm.compile(
    optimizer=Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_lstm.summary()

# --- 6. Wrap Generators ---
lstm_train_generator = lstm_data_generator(train_generator)
lstm_val_generator = lstm_data_generator(validation_generator)

# --- 7. Fix Infinite Loop by Setting Steps ---
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

# --- 8. Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(model_name, save_best_only=True)

# --- 9. TRAIN MODEL ---
print("\nStarting LSTM Training...")

history = model_lstm.fit(
    lstm_train_generator,
    epochs=epochs_max,
    steps_per_epoch=steps_per_epoch,
    validation_data=lstm_val_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, checkpoint]
)

print(f"\nTraining finished. Best model saved as {model_name}.")
