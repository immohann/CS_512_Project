# Import helper functions we're going to use
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
from keras.callbacks import CSVLogger

import seaborn as sns
import matplotlib.pyplot as plt    

train="data/data2a/training/"
test="data/data2a/validation/"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Create ImageDataGenerator training instance without data augmentation
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
# Import data from directories and turn it into batches
train_data = train_datagen.flow_from_directory(train,
                                               batch_size=32, # number of images to process at a time 
                                               target_size=(224, 224), # convert all images to be 224 x 224
                                               class_mode="categorical", # type of problem we're working on
                                               seed=42, shuffle=True)

valid_data = test_datagen.flow_from_directory(test,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="categorical",
                                               seed=42, shuffle=True)



model_list = ['Vgg16', 'Vgg19', 'Resnet']
noise_list = [0.01, 0.05]

for noise_level in noise_list:

    for model_chosen in model_list:
        
        if model_chosen == "Vgg19":
            pre_trained = VGG19(input_shape = [224, 224, 3], weights = 'imagenet', include_top = False) 
        
        elif model_chosen == "Vgg16":
    
            pre_trained = VGG16(input_shape = [224, 224, 3], weights = 'imagenet', include_top = False)  
        ###
        else:
            pre_trained = ResNet50(input_shape = [224, 224, 3], weights = 'imagenet', include_top = False) 

        filename = str(model_chosen) + "_" + str(noise_level) + "_damage_location" +".txt"
        f = open(filename, "w")

        # this will exclude the initial layers from training phase as there are already been trained.
        for layer in pre_trained.layers:
            layer.trainable = False


        new_model = tf.keras.Sequential()


        new_model.add(tf.keras.layers.GaussianNoise(noise_level))
        new_model.add(pre_trained)

        new_model.add(Flatten())
        new_model.add(Dense(128, activation = 'relu'))
        new_model.add(Dense(3, activation = 'softmax'))


        # x = Flatten()(pre_trained.output)
        # x = Dense(128, activation = 'relu')(x)   # we can add a new fully connected layer but it will increase the execution time.
        # x = Dense(3, activation = 'softmax')(x)  # adding the output layer with softmax function as this is a multi label classification problem.

        # model = Model(inputs = pre_trained.input, outputs = x)

        # model = Model(inputs = new_model.input, outputs = new_model.output)

        new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        filename_csv = str(model_chosen) + "_" + str(noise_level) + "_damage_location" +".csv"
        csv_logger = CSVLogger(filename_csv, append=True, separator=';')

        # Fit the model 
        history = new_model.fit(train_data, epochs=10, steps_per_epoch=len(train_data),
                                validation_data=valid_data,
                                validation_steps=len(valid_data), callbacks=[csv_logger])

        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # # Fit the model 
        # history = model.fit(train_data, epochs=10, steps_per_epoch=len(train_data),
        #                         validation_data=valid_data,
        #                         validation_steps=len(valid_data))


        # save_path = "saved_checkpoints/" + model_chosen + "car_damage_severity_model.h5"

    # model.save(save_path)

        save_path = "noise_saved_checkpoints/" + model_chosen + str(noise_level) + "car_damage_location_model.h5"

        new_model.save(save_path)


        test_data = test_datagen.flow_from_directory(test,
                                                batch_size=32,
                                                target_size=(224, 224),
                                                class_mode="categorical",
                                                seed=30, shuffle=False)


        prediction = new_model.predict(test_data)

        y_pred = []
        for each in prediction:
            max_value = max(each)

            max_index = list(each).index(max_value)

            y_pred.append(max_index)

        conf_matrix = confusion_matrix(y_pred=y_pred, y_true=valid_data.labels)

        ax= plt.subplot()
        sns.heatmap(conf_matrix, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix for' + str(model_chosen) +'with Gaussian Noise'); 
        ax.xaxis.set_ticklabels(['Front', 'Rear', 'Side']); ax.yaxis.set_ticklabels(['Front', 'Rear', 'Side'])

        save_path = str(model_chosen) + "_" + str(noise_level) + "_damage_location_noise_" + ".jpg"
        plt.savefig(save_path)


        print(" \t\t\t Model -" + str(model_chosen) + str(noise_level) + "with Gaussian Noise")
        print(classification_report(y_true=valid_data.labels, y_pred=y_pred))

        f.write(classification_report(y_true=valid_data.labels, y_pred=y_pred))

        f.close()