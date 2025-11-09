import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
import json
import os


data_dir = 'Sea'
img_height, img_width = 224, 224
batch_size = 32
num_classes = 24
epochs = 5  
base_dir = 'models_and_checkpoints'
os.makedirs(base_dir, exist_ok=True)

models_dir = os.path.join(base_dir, 'models')
os.makedirs(models_dir, exist_ok=True)


model_name = 'efficient_model_test'
model_dir = os.path.join(models_dir, model_name)
checkpoints_dir = os.path.join(model_dir, 'checkpoints')

os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

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


base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))


for layer in base_model.layers[-15:]: 
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.6)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)


def lr_schedule(epoch):
    initial_lr = 0.0001
    drop_factor = 0.5
    drop_every = 5
    if epoch % drop_every == 0 and epoch != 0:
        return initial_lr * (drop_factor ** (epoch // drop_every))
    return initial_lr


model.compile(optimizer=Adam(learning_rate=lr_schedule(0)),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


class CheckpointInfoSaver(Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model_checkpoints_dir = checkpoints_dir

    def on_epoch_end(self, epoch, logs=None):
        checkpoint_name = f"{self.model_name}_epoch_{epoch+1}.keras"
        checkpoint_path = os.path.join(self.model_checkpoints_dir, checkpoint_name)
        self.model.save(checkpoint_path)

     
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

        print(f"Checkpoint saved at epoch {epoch + 1}")


checkpoint_callback = CheckpointInfoSaver(model_name)

lr_scheduler_callback = LearningRateScheduler(lr_schedule)


history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint_callback, lr_scheduler_callback]
)


model_path = os.path.join(model_dir, f"{model_name}.keras")
model.save(model_path) 