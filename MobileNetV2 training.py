import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import json
import os

# Параметры
data_dir = 'Sea'
img_height, img_width = 224, 224
batch_size = 32
num_classes = 24
epochs = 25  # Количество эпох

# Папка для сохранения модели и чекпоинтов
base_dir = 'models_and_checkpoints'
os.makedirs(base_dir, exist_ok=True)

models_dir = os.path.join(base_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

# Название модели
model_name = 'MobileNetV2_model'
model_dir = os.path.join(models_dir, model_name)
checkpoints_dir = os.path.join(model_dir, 'checkpoints')

# Создаем папки для модели и чекпоинтов сразу
os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

# Генератор данных с аугментацией
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Загрузка данных с сохранением пропорций
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Используем MobileNetV2 без верхнего слоя
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Замораживаем базовые слои
for layer in base_model.layers:
    layer.trainable = False

# Добавляем новые слои
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

# Финальная модель
model = Model(inputs=base_model.input, outputs=output)

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Класс для сохранения полноценных чекпоинтов с моделью и дополнительной информацией
class CheckpointInfoSaver(Callback):
    def __init__(self, model_name, save_frequency=5):
        super().__init__()
        self.model_name = model_name
        self.best_val_loss = float('inf')  # Инициализация лучшей валидационной потери
        self.model_checkpoints_dir = checkpoints_dir  
        self.save_frequency = save_frequency  # Частота сохранения чекпоинтов
        self.last_saved_epoch = -1  # Последняя сохраненная эпоха

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get("val_loss")
        
        # Обновляем лучшую валидационную потерю только если прошли save_frequency эпох
        if (epoch - self.last_saved_epoch) >= self.save_frequency:
            self.best_val_loss = current_val_loss
            self.last_saved_epoch = epoch

                # Генерируем имя чекпоинта на основе наилучшей validation loss
            checkpoint_name = f"{self.model_name}_best_val_loss_{current_val_loss:.2f}_epoch_{epoch+1}.keras"
            checkpoint_path = os.path.join(self.model_checkpoints_dir, checkpoint_name)

                # Сохраняем всю модель
            self.model.save(checkpoint_path)

                # Сохраняем метаданные чекпоинта в JSON
            checkpoint_data = {
                "epoch": epoch + 1,
                "val_loss": logs["val_loss"],
                "val_accuracy": logs["val_accuracy"],
                "train_loss": logs["loss"],
                "train_accuracy": logs["accuracy"]
            }
            json_checkpoint_path = checkpoint_path.replace('.keras', '.json')
            with open(json_checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f)

            print(f"Checkpoint saved at epoch {epoch + 1} with improved val_loss: {current_val_loss:.2f}")

# Колбэк для сохранения чекпоинтов
checkpoint_callback = CheckpointInfoSaver(model_name, save_frequency=5)


# Обучаем модель
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint_callback]
)

# Сохраняем итоговую модель
model_path = os.path.join(model_dir, f"{model_name}.keras")
model.save(model_path)
