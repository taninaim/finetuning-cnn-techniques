from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        'hot-dog-not-hot-dog/train',
        target_size=(256, 256),
        batch_size=64,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        'hot-dog-not-hot-dog/test',
        target_size=(256, 256),
        batch_size=64,
        class_mode='categorical')

from keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
    if epoch <= 25:
        return 0.001
    else:
        return 0.0001

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
callbacks = [lr_scheduler]

r = model.fit_generator(
        train_generator,
        steps_per_epoch=(train_generator.n)//32,
        epochs=50,
        validation_data=test_generator,
        validation_steps=(test_generator.n)//32,
        callbacks=callbacks)

import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')

plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()

plt.savefig("hot-dog.png")
