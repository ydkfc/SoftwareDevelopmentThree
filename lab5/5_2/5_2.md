```python
import os
import zipfile

local_zip = 'do/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('do/')
zip_ref.close()

local_zip = 'do/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('do/')
zip_ref.close()

```


```python
rock_dir = os.path.join('do/rps/rock')
paper_dir = os.path.join('do/rps/paper')
scissors_dir = os.path.join('do/rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

```

    total training rock images: 840
    total training paper images: 840
    total training scissors images: 840
    ['rock04-047.png', 'rock07-k03-036.png', 'rock04-070.png', 'rock07-k03-000.png', 'rock02-116.png', 'rock04-107.png', 'rock03-060.png', 'rock03-058.png', 'rock05ck01-069.png', 'rock03-004.png']
    ['paper02-048.png', 'paper06-055.png', 'paper07-116.png', 'paper06-119.png', 'paper05-043.png', 'paper06-017.png', 'paper05-046.png', 'paper04-006.png', 'paper02-037.png', 'paper01-083.png']
    ['scissors03-008.png', 'testscissors01-113.png', 'scissors02-117.png', 'scissors03-061.png', 'testscissors03-045.png', 'scissors03-058.png', 'scissors02-074.png', 'testscissors02-008.png', 'scissors01-011.png', 'scissors04-031.png']



```python
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2

next_rock = [os.path.join(rock_dir, fname) 
                for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) 
                for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()
```


    
![png](1_files/1_2_0.png)
    



    
![png](1_files/1_2_1.png)
    



    
![png](1_files/1_2_2.png)
    



    
![png](1_files/1_2_3.png)
    



    
![png](1_files/1_2_4.png)
    



    
![png](1_files/1_2_5.png)
    



```python
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "do/rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "do/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("rps.h5")

```

    2023-05-31 05:37:38.363730: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2023-05-31 05:37:38.363768: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.


    Found 2520 images belonging to 3 classes.
    Found 372 images belonging to 3 classes.


    2023-05-31 05:37:50.867030: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
    2023-05-31 05:37:50.867069: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
    2023-05-31 05:37:50.867099: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (codespaces-756f23): /proc/driver/nvidia/version does not exist
    2023-05-31 05:37:50.867765: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 148, 148, 64)      1792      
                                                                     
     max_pooling2d (MaxPooling2D  (None, 74, 74, 64)       0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 72, 72, 64)        36928     
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 36, 36, 64)       0         
     2D)                                                             
                                                                     
     conv2d_2 (Conv2D)           (None, 34, 34, 128)       73856     
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 17, 17, 128)      0         
     2D)                                                             
                                                                     
     conv2d_3 (Conv2D)           (None, 15, 15, 128)       147584    
                                                                     
     max_pooling2d_3 (MaxPooling  (None, 7, 7, 128)        0         
     2D)                                                             
                                                                     
     flatten (Flatten)           (None, 6272)              0         
                                                                     
     dropout (Dropout)           (None, 6272)              0         
                                                                     
     dense (Dense)               (None, 512)               3211776   
                                                                     
     dense_1 (Dense)             (None, 3)                 1539      
                                                                     
    =================================================================
    Total params: 3,473,475
    Trainable params: 3,473,475
    Non-trainable params: 0
    _________________________________________________________________


    2023-05-31 05:37:52.133529: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 34020000 exceeds 10% of free system memory.


    Epoch 1/25


    2023-05-31 05:37:54.591454: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 34020000 exceeds 10% of free system memory.
    2023-05-31 05:37:54.639747: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 706535424 exceeds 10% of free system memory.
    2023-05-31 05:37:55.390346: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 176633856 exceeds 10% of free system memory.
    2023-05-31 05:37:55.505956: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 167215104 exceeds 10% of free system memory.


    20/20 [==============================] - 76s 4s/step - loss: 2.2581 - accuracy: 0.3512 - val_loss: 1.0885 - val_accuracy: 0.3333
    Epoch 2/25
    20/20 [==============================] - 72s 4s/step - loss: 1.1346 - accuracy: 0.4159 - val_loss: 1.0865 - val_accuracy: 0.3333
    Epoch 3/25
    20/20 [==============================] - 71s 4s/step - loss: 1.0701 - accuracy: 0.4452 - val_loss: 0.9181 - val_accuracy: 0.6962
    Epoch 4/25
    20/20 [==============================] - 73s 4s/step - loss: 1.0473 - accuracy: 0.5194 - val_loss: 0.9639 - val_accuracy: 0.6075
    Epoch 5/25
    20/20 [==============================] - 72s 4s/step - loss: 0.9400 - accuracy: 0.5639 - val_loss: 0.5619 - val_accuracy: 0.9731
    Epoch 6/25
    20/20 [==============================] - 75s 4s/step - loss: 0.7896 - accuracy: 0.6310 - val_loss: 0.4617 - val_accuracy: 0.9570
    Epoch 7/25
    20/20 [==============================] - 73s 4s/step - loss: 0.6588 - accuracy: 0.6909 - val_loss: 0.2271 - val_accuracy: 0.9651
    Epoch 8/25
    20/20 [==============================] - 73s 4s/step - loss: 0.6490 - accuracy: 0.7353 - val_loss: 0.6063 - val_accuracy: 0.6747
    Epoch 9/25
    20/20 [==============================] - 72s 4s/step - loss: 0.4745 - accuracy: 0.8044 - val_loss: 0.0833 - val_accuracy: 0.9919
    Epoch 10/25
    20/20 [==============================] - 72s 4s/step - loss: 0.4762 - accuracy: 0.8115 - val_loss: 0.1124 - val_accuracy: 0.9677
    Epoch 11/25
    20/20 [==============================] - 74s 4s/step - loss: 0.3338 - accuracy: 0.8595 - val_loss: 0.1038 - val_accuracy: 1.0000
    Epoch 12/25
    20/20 [==============================] - 72s 4s/step - loss: 0.2933 - accuracy: 0.8921 - val_loss: 0.0606 - val_accuracy: 0.9919
    Epoch 13/25
    20/20 [==============================] - 72s 4s/step - loss: 0.2438 - accuracy: 0.9028 - val_loss: 0.8510 - val_accuracy: 0.6102
    Epoch 14/25
    20/20 [==============================] - 72s 4s/step - loss: 0.2703 - accuracy: 0.8948 - val_loss: 0.2481 - val_accuracy: 0.9731
    Epoch 15/25
    20/20 [==============================] - 72s 4s/step - loss: 0.1673 - accuracy: 0.9472 - val_loss: 0.0454 - val_accuracy: 0.9892
    Epoch 16/25
    20/20 [==============================] - 72s 4s/step - loss: 0.2209 - accuracy: 0.9198 - val_loss: 0.0516 - val_accuracy: 0.9919
    Epoch 17/25
    20/20 [==============================] - 72s 4s/step - loss: 0.1679 - accuracy: 0.9361 - val_loss: 0.0545 - val_accuracy: 0.9785
    Epoch 18/25
    20/20 [==============================] - 72s 4s/step - loss: 0.1289 - accuracy: 0.9571 - val_loss: 0.0157 - val_accuracy: 1.0000
    Epoch 19/25
    20/20 [==============================] - 71s 4s/step - loss: 0.1863 - accuracy: 0.9333 - val_loss: 0.0281 - val_accuracy: 1.0000
    Epoch 20/25
    20/20 [==============================] - 72s 4s/step - loss: 0.1287 - accuracy: 0.9528 - val_loss: 0.0077 - val_accuracy: 1.0000
    Epoch 21/25
    20/20 [==============================] - 71s 4s/step - loss: 0.1098 - accuracy: 0.9627 - val_loss: 0.0373 - val_accuracy: 0.9866
    Epoch 22/25
    20/20 [==============================] - 72s 4s/step - loss: 0.1285 - accuracy: 0.9532 - val_loss: 0.0507 - val_accuracy: 0.9731
    Epoch 23/25
    20/20 [==============================] - 73s 4s/step - loss: 0.1657 - accuracy: 0.9429 - val_loss: 0.0197 - val_accuracy: 1.0000
    Epoch 24/25
    20/20 [==============================] - 72s 4s/step - loss: 0.0554 - accuracy: 0.9841 - val_loss: 0.0045 - val_accuracy: 1.0000
    Epoch 25/25
    20/20 [==============================] - 72s 4s/step - loss: 0.1476 - accuracy: 0.9437 - val_loss: 0.0715 - val_accuracy: 0.9677



```python
import matplotlib.pyplot as plt
```


```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
```


    
![png](1_files/1_5_0.png)
    



    <Figure size 640x480 with 0 Axes>

