# Import helper functions we're going to use
from gc import callbacks
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir

# importing the libraries
from keras.models import Model
from keras.layers import Flatten, Dense
#import VGG16
from keras.applications.vgg16 import VGG16
#import VGG19
from keras.applications.vgg19 import VGG19
#import ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.preprocessing import image 
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt    

from keras.callbacks import CSVLogger

train="data/data1a/training"
test="data/data1a/validation"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Create ImageDataGenerator training instance without data augmentation
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
# Import data from directories and turn it into batches
train_data = train_datagen.flow_from_directory(train,
                                               batch_size=32, # number of images to process at a time 
                                               target_size=(224, 224), # convert all images to be 224 x 224
                                               class_mode="binary", # type of problem we're working on
                                               seed=42, subset='training' ,shuffle = True )

valid_data = test_datagen.flow_from_directory(test,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42,shuffle = True )



model_list = ['Vgg16', 'Vgg19', 'Resnet']

noise_list = [0.01, 0.05]

for noise_level in noise_list:

    for model_chosen in model_list:
        
        if model_chosen == "Vgg16":
            base_model = tf.keras.applications.VGG16(weights='imagenet',include_top=False,input_shape=(224, 224, 3))

        
        elif model_chosen == "Vgg19":
    
            base_model = tf.keras.applications.VGG19( weights='imagenet',include_top=False,input_shape=(224, 224, 3))
        else:
            base_model = ResNet50(weights='imagenet',include_top=False,input_shape=(224, 224, 3))


        filename = str(model_chosen) + "_" + str(noise_level) + "damage_or_not_" +  ".txt"
        f = open(filename, "w")

        for layer in base_model.layers:
            layer.trainable = False

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.GaussianNoise(noise_level))
        model.add(base_model)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # model.layers[1].trainable = False

        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy'])

        filename_csv = str(model_chosen) + "_" + str(noise_level) + "damage_or_not_" +  ".csv"
        csv_logger = CSVLogger(filename_csv, append = True, separator = ";")
        # Fit the model 
        history = model.fit(train_data, epochs=10, steps_per_epoch=len(train_data),
                                validation_data=valid_data,
                                validation_steps=len(valid_data), callbacks = [csv_logger])

        # f.write(history)
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # # Fit the model 
        # history = model.fit(train_data, epochs=10, steps_per_epoch=len(train_data),
        #                         validation_data=valid_data,
        #                         validation_steps=len(valid_data))


        # save_path = "saved_checkpoints/" + model_chosen + "car_damage_severity_model.h5"

        save_path = "noise_saved_checkpoints/" + model_chosen + str(noise_level) + "_car_damage_or_not_detection-model.h5"

        model.save(save_path)


        test_data = test_datagen.flow_from_directory(test,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42,shuffle = False )



        prediction = model.predict(test_data)


        y_pred = []
        for each in prediction:
            y_pred.append(np.round(each.item()))

        conf_matrix = confusion_matrix(y_pred=y_pred, y_true=valid_data.labels)

        ax= plt.subplot()
        sns.heatmap(conf_matrix, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix for VGG 16'); 
        ax.xaxis.set_ticklabels(['Damage', 'Whole']); ax.yaxis.set_ticklabels(['Damage', 'Whole'])
        save_path = str(model_chosen) + "damage_or_not_" + str(noise_level) + ".jpg"
        plt.savefig(save_path)


        print(" \t\t\t Model -" + str(model_chosen) + str(noise_level) + "with Gaussian Noise")
        print(classification_report(y_true=valid_data.labels, y_pred=y_pred))

        f.write(classification_report(y_true=valid_data.labels, y_pred=y_pred))

        f.close()


        # print("Done")