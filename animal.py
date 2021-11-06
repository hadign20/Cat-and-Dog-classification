
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.utils import plot_model
from keras import backend as K

from keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau


'''
========================================
initializations
========================================
'''
save_dir = 'G:/My Drive/Sem2/MachineLearning/Project/animal/save_dir/models/'

train_dir = 'G:/My Drive/Sem2/MachineLearning/Project/data/catdog/train'
valid_dir = 'G:/My Drive/Sem2/MachineLearning/Project/data/catdog/validation'
test_dir = 'G:/My Drive/Sem2/MachineLearning/Project/data/catdog/test'

train_size = 2000
valid_size = 1000
test_size = 20
epochs = 20
batch_size = 8
learning_rate = 1e-4
beta1 = 0.9
lr_decay = 0.5
opt = "rmsprop" # adam or sgd or rmsprop


'''
========================================
general
========================================
'''
img_height = 150
img_width = 150
img_channels = 3

if opt == "adam":
    optim = optimizers.Adam(lr=learning_rate, beta_1=beta1, decay=lr_decay, amsgrad=False)
elif opt == "sgd":
    optim = optimizers.SGD(lr=1e-4, momentum=0.9)
elif opt== 'rmsprop':
    optim = 'rmsprop'

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def plot_charts(trained_model):
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 8))
    plt.plot(trained_model.history["loss"], label="train_loss")
    plt.plot(trained_model.history["val_loss"], label="val_loss")
    plt.plot(trained_model.history["acc"], label="train_acc")
    plt.plot(trained_model.history["val_acc"], label="val_acc")
    plt.plot(np.argmin(trained_model.history["val_loss"]), np.min(trained_model.history["val_loss"]), marker="x", color="r", label="best model")
    plt.title("training charts")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")

'''
========================================
model
========================================
'''

''' vgg
model = Sequential([
Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
Conv2D(64, (3, 3), activation='relu', padding='same'),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(128, (3, 3), activation='relu', padding='same'),
Conv2D(128, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(256, (3, 3), activation='relu', padding='same',),
Conv2D(256, (3, 3), activation='relu', padding='same',),
Conv2D(256, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Flatten(),
Dense(4096, activation=’relu’),
Dense(4096, activation=’relu’),
Dense(1000, activation=’softmax’)
])
'''

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape, name='c1'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), name='c2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), name='c3'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer= optim, loss='binary_crossentropy', metrics=['accuracy'])

plot_model(model, to_file=os.path.join(save_dir +"model.png"))
if os.path.exists(os.path.join(save_dir +"model.txt")):
    os.remove(os.path.join(save_dir +"model.txt"))
with open(os.path.join(save_dir +"model.txt"),'w') as fh:
    model.summary(positions=[.3, .55, .67, 1.], print_fn=lambda x: fh.write(x + '\n'))

model.summary()


'''
========================================
training
========================================
'''

callbacks = [
    EarlyStopping(patience=1000, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint(save_dir + 'model.{epoch:02d}-{val_loss:.2f}.h5', verbose=1, save_best_only=True, save_weights_only=True)
]


train_aug = ImageDataGenerator(horizontal_flip=True, shear_range=0.3, rotation_range = 0.3, zoom_range=0.2, rescale=1. / 255)
test_aug = ImageDataGenerator(rescale=1. / 255)

generate_data_train = train_aug.flow_from_directory(train_dir, batch_size=batch_size, target_size=(img_height, img_width), class_mode='binary')

generate_data_valid = test_aug.flow_from_directory(valid_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary')



H = model.fit_generator(generate_data_train, 
                        epochs= epochs,
                        samples_per_epoch = train_size,
                        steps_per_epoch = 1000/batch_size, 
                        validation_data = generate_data_valid,
                        callbacks = callbacks,
                        validation_steps= 1000/batch_size*2)

plot_charts(H)



H = model.load_weights(save_dir + 'model.36-0.50.h5')
print("\n***************\nmodel loaded.\n")
model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])

valid_images = [".jpg",".gif",".png",".tga"]
test_images = os.listdir(test_dir)
#test_images = Path(test_dir)
for i in test_images:
    ext = os.path.splitext(i)[1]
    if ext.lower() not in valid_images:
        continue
    i = os.path.join(test_dir, i )
    test_image = image.load_img(i, target_size = (150,150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis = 0)
    img=mpimg.imread(i)
    imgplot = plt.imshow(img)
    plt.show()
    prediction = model.predict(test_image)
    generate_data_train.class_indices
    if prediction[0][0] >= 0.5:
        output = "dog"
    else:
        output = "cat"
    print(" predicted as: ", output)
    

K.clear_session()
del model











