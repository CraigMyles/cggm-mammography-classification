import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import math
import pydicom
import cv2
import random
random.seed(10)

import keras
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import classification_report, roc_auc_score


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('--manifest_path', type=dir_path, required=True)
    parser.add_argument('--metadata_path', type=argparse.FileType('r'), required=True)
    return parser.parse_args()


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir: {path} is not a valid path")

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle

def metadata_preprocessing(path_to_metadata,path_to_clinical_data, manifest_path):
    # Create Pandas dataframes for both clinical data file and metadata file
    clinical_data = pd.read_excel(path_to_clinical_data)
    meta_data = pd.read_csv(path_to_metadata)

    # Subset into only SubjectID, Number of Images, and File Location
    meta_subset = meta_data.loc[:, ['Subject ID','Number of Images', 'File Location']]
    
    # Align column name
    df1 = clinical_data.rename(columns={"ID1": "Subject_ID"})
    df2 = meta_subset.rename(columns={"Subject ID": "Subject_ID"})
    # Merge the two dataframes, using _ID column as key
    df3 = pd.merge(df1, df2, on = 'Subject_ID')
    df3.set_index('Subject_ID', inplace = True)

    # Add column for image 1 and image 2
    df3['img_1'] = ''
    df3['img_2'] = ''

    #Sort for cases where single patient_id's have >2 mammography images
    print("Sorting for cases where single patient_id's have >2 mammography images...")
    for i in tqdm(range(len(df3))):
        if i==0:
            continue
        #If the current file path is the same as the previous path
        if df3.iloc[i]['File Location'] == df3.iloc[i-1]['File Location']:
            #If file path is the same as previous, that means there is >2
            df3.iloc[i, df3.columns.get_loc('img_1')] = "1-3.dcm"
            df3.iloc[i, df3.columns.get_loc('img_2')] = "1-4.dcm"
        else:
            df3.iloc[i, df3.columns.get_loc('img_1')] = "1-1.dcm"
            df3.iloc[i, df3.columns.get_loc('img_2')] = "1-2.dcm"

    #Create new empty dataframe
    df4 = pd.DataFrame(columns=["subject_id", "leftright", "age", "abnormality",
                                "classification", "subtype", "file_location"])
    #fix index, one to right.
    df3 = df3.reset_index()

    # Iterate through each line in the dataframe, determine file
    # location based on odd/even integers for scan type. (see pdf
    # for data stratificaiton explanation)
    def create_row(i, df4, flag=False):
        appended_data = []
        j = 0
        while j < 2:

            if not flag:
                if j == 0:
                    file_loc = str(df3.iloc[i, df3.columns.get_loc('File Location')])+"/1-1.dcm"
                else:
                    file_loc = str(df3.iloc[i, df3.columns.get_loc('File Location')])+"/1-2.dcm"
            
            if flag:
                if j == 0:
                    file_loc = str(df3.iloc[i, df3.columns.get_loc('File Location')])+"/1-3.dcm"
                else:
                    file_loc = str(df3.iloc[i, df3.columns.get_loc('File Location')])+"/1-4.dcm"
    # Uncomment if debugging
    #         print("iteration:"+str(i))
    #         print(file_loc)
            new_row = {
                'subject_id':    df3.iloc[i, df3.columns.get_loc('Subject_ID')],
                'leftright':     df3.iloc[i, df3.columns.get_loc('LeftRight')],
                'age':           df3.iloc[i, df3.columns.get_loc('Age')],
                'abnormality':   df3.iloc[i, df3.columns.get_loc('abnormality')],
                'classification':df3.iloc[i, df3.columns.get_loc('classification')],
                'subtype':       df3.iloc[i, df3.columns.get_loc('subtype')],
                'file_location': file_loc
            }
            appended_data.append(new_row)
            df4 = df4.append(new_row, ignore_index=True)
    #         print(len(df4))
            j += 1
        return appended_data
    
    #For all items in the manifest
    print("Preprocessing for all items in the manifest...")
    for i in tqdm(range(len(df3))):
        #skip 0th item because cant compare to -1th item.
        if i==0:
            #create regular row
            data_to_append = create_row(i, df4)
            
            df4 = df4.append(data_to_append, ignore_index=True)
            continue
        #if the file location equals the same as the one before...
        if df3.iloc[i]['File Location'] == df3.iloc[i-1]['File Location']:
            #True because this folder has 1,2,3,4 images
            data_to_append = create_row(i, df4, True)
            
            df4 = df4.append(data_to_append, ignore_index=True)
        else:
            data_to_append = create_row(i, df4)
            
            df4 = df4.append(data_to_append, ignore_index=True)

    df4.to_csv("./CMMD_metadata_subset.csv", index=False)

    #Append the /path/to/manifest/ to "1-1.dcm" or "1-2.dcm" etc...
    print("Append the /path/to/manifest/ to 1-1.dcm or 1-2.dcm etc...")
    for i in tqdm(range(len(df4))):
        begin_path = manifest_path[:-1]
        acc_file = df4.iloc[i]['file_location']
        my_file_loc = str(begin_path+acc_file[1:])
        if not os.path.isfile(my_file_loc):
            print("WARNING, the following file does not exist:\n"+my_file_loc)

    #Save newly curated metadata file.
    # df4.to_csv('CMMD_metadata.csv', encoding='utf-8')
    return df4

def stratify_data(df, manifest_path):
    #Create dataframe which excludes all non benign classifications
    benign_df = df.loc[df['classification'] == 'Benign']

    #Create dataframe which excludes all non malignant classifications
    malignant_df = df.loc[df['classification'] == 'Malignant']

    cmmd_dir = manifest_path[:-23]
    benign_loc = cmmd_dir+"cmmd_data/benign/"
    malignant_loc = cmmd_dir+"cmmd_data/malignant/"

    #create directory if doesnt exist
    Path(benign_loc).mkdir(parents=True, exist_ok=True)
    #create directory if doesnt exist
    Path(malignant_loc).mkdir(parents=True, exist_ok=True)      

    matches = ["1-3.dcm", "1-4.dcm"]

    def create_benign_malignant(df, dest_folder):
        
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):  
            src = cmmd_dir+"manifest-1616439774456/"+row['file_location']
            basename = os.path.basename(src) #<- basename = file name + extension
            if any(x in basename for x in matches): # Check for a 3rd or 4th in path
                #append "_b" to subject ID to show this is a second case for the same patient
                dest = dest_folder+row['subject_id']+"_b/"+basename 
                Path(dest_folder+row['subject_id']+"_b/").mkdir(parents=True, exist_ok=True)
            else:
                dest = dest_folder+row['subject_id']+"/"+basename
                Path(dest_folder+row['subject_id']+"/").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)

    print("Building benign DICOM collection")
    create_benign_malignant(benign_df, benign_loc)
    print("Building malignant DICOM collection")
    create_benign_malignant(malignant_df, malignant_loc)

    #method to move 20% of a directory into another location
    #splits a dataset into train/test and/or train/validate
    def create_test_dataset(data_location, destination):
        count = (len(os.listdir(data_location))/5)
        count = (math.ceil(count))
        test_set = random.sample(os.listdir(data_location), count)
        

        for i in tqdm(range(len(test_set))):

            shutil.move(data_location+test_set[i], destination)

    benign_testset_location = cmmd_dir+"TEST/benign/"
    malignant_testset_location = cmmd_dir+"TEST/malignant/"
    #create directory if doesnt exist
    Path(benign_testset_location).mkdir(parents=True, exist_ok=True)
    #create directory if doesnt exist
    Path(malignant_testset_location).mkdir(parents=True, exist_ok=True)   
        
    print("Creating 20% test split for benign set...")        
    create_test_dataset(benign_loc, benign_testset_location)

    print("Creating 20% test split for malignant set...")  
    create_test_dataset(malignant_loc, malignant_testset_location)

    benign_valset_location = cmmd_dir+"VAL/benign/"
    malignant_valset_location = cmmd_dir+"VAL/malignant/"
    #create directory if doesnt exist
    Path(benign_valset_location).mkdir(parents=True, exist_ok=True)
    #create directory if doesnt exist
    Path(malignant_valset_location).mkdir(parents=True, exist_ok=True)

    print("Creating 20% validation split for benign set...")       
    create_test_dataset(benign_loc, benign_valset_location)
    print("Creating 20% validation split for malignant set...")  
    create_test_dataset(malignant_loc, malignant_valset_location)

    shutil.move(cmmd_dir+"cmmd_data", cmmd_dir+"TRAIN")

    def move_dcm_from_subdir(source, destination):
        Path(destination).mkdir(parents=True, exist_ok=True)
        files_list = os.listdir(source)
        j=1
        for files in files_list:
            files_list2 = os.listdir(source+files)
            for x in files_list2:
                shutil.move(source+files+"/"+x, destination+str(j)+".dcm")
                j+=1
            
    move_dcm_from_subdir(source=cmmd_dir+"TRAIN/benign/",
                        destination = cmmd_dir+"cmmd_data/TRAIN/benign/")
    move_dcm_from_subdir(source=cmmd_dir+"TRAIN/malignant/",
                        destination = cmmd_dir+"cmmd_data/TRAIN/malignant/")

    move_dcm_from_subdir(source=cmmd_dir+"VAL/benign/",
                        destination = cmmd_dir+"cmmd_data/VAL/benign/")
    move_dcm_from_subdir(source=cmmd_dir+"VAL/malignant/",
                        destination = cmmd_dir+"cmmd_data/VAL/malignant/")

    move_dcm_from_subdir(source=cmmd_dir+"TEST/benign/",
                        destination = cmmd_dir+"cmmd_data/TEST/benign/")
    move_dcm_from_subdir(source=cmmd_dir+"TEST/malignant/",
                        destination = cmmd_dir+"cmmd_data/TEST/malignant/")

    def rm_dir(directiory):
        ## Try to remove tree
        try:
            shutil.rmtree(directiory)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))

    rm_dir(cmmd_dir+"TRAIN")
    rm_dir(cmmd_dir+"TEST")
    rm_dir(cmmd_dir+"VAL")


    def convert_dicom_to_png(input_dir, output_dir):
        
        if not os.path.exists(output_dir): #if file doesnt exist, create it
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
        classification_list = [ classification for classification in  os.listdir(input_dir)]
        for classification in classification_list:
            print("Working on "+classification+" for \n"+output_dir)
            dicom_list = [ dcm_image for dcm_image in  os.listdir(input_dir+classification)]
            for dcm_image in tqdm(dicom_list):
                ds = pydicom.read_file(input_dir+classification+"/"+dcm_image) # read dicom image
                img = ds.pixel_array # get image array
                if not os.path.exists(output_dir + classification): #if file doesnt exist, create it
                    Path(output_dir + classification).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(output_dir + classification + "/img_" +dcm_image.replace('.dcm','.png'),img) # write png image


    convert_dicom_to_png(input_dir = cmmd_dir + 'cmmd_data/TRAIN/',
                        output_dir = cmmd_dir + 'cmmd_data/PNG/TRAIN/')    
        
    convert_dicom_to_png(input_dir = cmmd_dir + 'cmmd_data/TEST/',
                        output_dir = cmmd_dir + 'cmmd_data/PNG/TEST/')    

    convert_dicom_to_png(input_dir = cmmd_dir + 'cmmd_data/VAL/',
                        output_dir = cmmd_dir + 'cmmd_data/PNG/VAL/') 

    #Tidy up directory
    rm_dir(cmmd_dir+"cmmd_data/TRAIN/")
    rm_dir(cmmd_dir+"cmmd_data/TEST/")
    rm_dir(cmmd_dir+"cmmd_data/VAL/")
    rm_dir(manifest_path)
    print("Data converted to PNG and filesystem is tidied.")

    return

def data_load_aug(cmmd_data_dir):

    cmmd_dir = cmmd_data_dir

    train_datagen = ImageDataGenerator(
            # rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=tf.keras.applications.xception.preprocess_input
    )
    test_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.xception.preprocess_input
    )
    train_generator = train_datagen.flow_from_directory(
            cmmd_dir+"PNG/TRAIN/",
            target_size=(299, 299),
            batch_size=32,
            shuffle=True,
            class_mode='binary',
            seed=14)
    validation_generator = test_datagen.flow_from_directory(
            cmmd_dir+"PNG/VAL/",
            target_size=(299, 299),
            batch_size=32,
            shuffle=True,
            class_mode='binary',
            seed=14)

    return train_generator, validation_generator

def xception_transfer_learning(train_generator, validation_generator):


    base_model = keras.applications.Xception(
        weights='imagenet',
        input_shape=(299, 299, 3),
        include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # fully connected layer
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    callbacks = [
        # keras.callbacks.ModelCheckpoint("/media/craig/Henry/ML_models/xception_aug/save_at_{epoch}.h5"),
        keras.callbacks.ModelCheckpoint('best_xception_model.h5', monitor='val_loss', mode='min', save_best_only=True),
        # keras.callbacks.TensorBoard(run_logdir),
        keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    ]

    model.compile(optimizer=keras.optimizers.Adam(),
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[keras.metrics.BinaryAccuracy()])
    history = model.fit(train_generator, epochs=500, callbacks=callbacks, validation_data=validation_generator)


    loss_hist = history.history['val_loss']

    lowest_val_loss = np.min(loss_hist)
    best_epochs = np.argmin(loss_hist) + 1
    print("The lowest validation loss is: "+str(lowest_val_loss)+"\nAt epoch: "+str(best_epochs))

    bin_acc_hist = history.history['val_binary_accuracy']

    lowest_val_loss = np.max(bin_acc_hist)
    best_epochs = np.argmax(bin_acc_hist) + 1
    print("The highest accuracy is: "+str(lowest_val_loss)+"\nAt epoch: "+str(best_epochs))

    return

def xception_fine_tune(train_generator, validation_generator):
    #load model
    model = keras.models.load_model("best_xception_model.h5")
    model.trainable = True

    callbacks = [
    # keras.callbacks.ModelCheckpoint("/media/craig/Henry/ML_models/xception_aug/save_at_{epoch}.h5"),
    keras.callbacks.ModelCheckpoint('best_xception_fine_tuned_model.h5', monitor='val_loss', mode='min', save_best_only=True),
    # keras.callbacks.TensorBoard(run_logdir),
    keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    ]

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.000001),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
    history = model.fit(train_generator, epochs=500, callbacks=callbacks, validation_data=validation_generator)
    return

def xception_fine_tune_test(cmmd_data_dir):

    def test_model(model):        
        predictions = model.predict(test_generator)
        class_one = predictions > 0.5
        acc = np.mean(class_one == test_generator.classes)
        print("Accuracy: "+str(acc))
        y_true = test_generator.classes
        auc_score = roc_auc_score(y_true, predictions)
        print("AUC:"+ str(auc_score))
        model_eval = model.evaluate(test_generator)
        predictions = (model.predict(test_generator) > 0.5).astype("int32")
        predictions = predictions.reshape(1,-1)[0]
        print(classification_report(y_true, predictions, target_names = ['Malignant (Class 0)','Benign (Class 1)']))

    print("Xception (Fine Tune)")

    test_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.xception.preprocess_input
    )

    test_generator = test_datagen.flow_from_directory(
            cmmd_data_dir+"PNG/TEST/",
            color_mode='rgb',
            target_size=(299, 299),
            batch_size=1,
            shuffle=False,
            class_mode='binary',
            seed=11)

    test_model(model=keras.models.load_model("best_xception_fine_tuned_model.h5"))
    return

def main():
    #Parse script args
    parsed_args = parse_arguments()

    #Get important variables from inputs
    manifest_path = parsed_args.manifest_path
    metadata_path = parsed_args.metadata_path
    cmmd_data_dir = manifest_path[:-23]+"cmmd_data/"

    #Path variables for passing
    path_to_metadata = manifest_path+"metadata.csv"
    path_to_clinical_data = metadata_path.name

    print("Preprocessing...")
    curated_metadata = metadata_preprocessing(path_to_metadata,path_to_clinical_data,manifest_path)

    print("Data stratification...")
    stratify_data(curated_metadata, manifest_path)

    print("Initalising data generators...")
    train_generator, validation_generator = data_load_aug(cmmd_data_dir)

    print("Training Xception model...")
    xception_transfer_learning(train_generator, validation_generator)

    print("Fine-Tuning Xception model...")
    xception_fine_tune(train_generator, validation_generator)

    print("Testing fine-tuned model...")
    xception_fine_tune_test(cmmd_data_dir)

if __name__ == "__main__":
    main()