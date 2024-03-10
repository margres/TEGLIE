import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from . import settings
from . import gen_cutouts, findlenses, preprocessing, models, utils
import tensorflow as tf
import os
import random
from astropy.io import fits

abs_p=  os.path.abspath(__file__)

# Function to create ground truth and predicted labels for binary classification
def get_y_true_pred_from_idx(idx_pred_1, idx_pred_0, n_lens, n_bogus):
    """
    Parameters:
    - idx_pred_1: Indices of predicted lenses
    - idx_pred_0: Indices of predicted bogus (non-lenses)
    - n_lens: Number of actual lenses
    - n_bogus: Number of actual bogus samples

    Output:
    - y_true: Ground truth labels (1 for lenses, 0 for bogus)
    - y_pred: Predicted labels based on provided indices
    """
    ## Lens
    y_true_1 = np.ones(n_lens)
    y_pred_1 = np.zeros_like(y_true_1)
    y_pred_1[idx_pred_1] = 1

    ## Bogus
    y_true_0 = np.zeros(n_bogus)
    y_pred_0 = y_true_0.copy()
    y_pred_0[idx_pred_0] = 1

    y_true = np.hstack((y_true_1, y_true_0))
    y_pred = np.hstack((y_pred_1, y_pred_0))
    return y_true, y_pred


# Function to create a backup DataFrame with info of training and testing
def df_backup_traintest(df, type_t=None):
    """
    Parameters:
    - df: DataFrame containing data (columns: KIDS_TILE, KIDS_ID, LABEL, FOLDER)
    - type_t: Type of the DataFrame (train or test)

    Output:
    - cc: New DataFrame with selected columns for backup
    """
    df_to_save = pd.DataFrame()
    df_to_save['KIDS_TILE'] = df['KIDS_TILE']
    df_to_save['KIDS_ID'] = df['KIDS_ID']
    df_to_save['LABEL'] = df['LABEL']
    df_to_save['type'] = type_t
    df_to_save['FOLDER'] = df['FOLDER']
    return df_to_save


# Function to retrain a model using data augmentation
def re_train(X_train, Y_train, X_test, Y_test, model=models.lens15(), folder=None, epochs=50):
    """
    Parameters:
    - X_train, Y_train: Training data and labels
    - X_test, Y_test: Testing data and labels
    - model: TensorFlow/Keras model to be trained
    - folder: Name of the folder for saving weights and results
    - epochs: Number of training epochs

    Output:
    - Trained model with saved weights
    """
    # NB!! Preprocessing
    if folder is None:
        raise ValueError('Folder name is needed!')
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=(-0.05, 0.05),
        height_shift_range=(-0.05, 0.05),
        horizontal_flip=True,
        vertical_flip=True
    )
    datagen.fit(X_train)

    # Callbacks for early stopping and saving the best model
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
    mcp_save = tf.keras.callbacks.ModelCheckpoint(
        os.path.join('Retrain', folder, 'weights.h5'),
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    # Training the model with data augmentation
    model.fit(
        datagen.flow(X_train, Y_train),
        epochs=epochs,
        validation_data=datagen.flow(X_test, Y_test),
        callbacks=[callback, mcp_save],
        shuffle=True,
        verbose=1
    )

# Function to split the dataset into training, testing, and validation sets
def train_test_val(df_X_0, df_X_1, folder=None):
    """
    Parameters:
    - df_X_0: DataFrame for class 0 (non-lenses)
    - df_X_1: DataFrame for class 1 (lenses)
    - folder: Name of the folder for saving dataframes and models

    Output:
    - X_train, X_test, X_val: Training, testing, and validation datasets
    - Y_train, Y_test, Y_val: Labels for training, testing, and validation datasets
    - df_X_val: DataFrame for the validation set
    """
    df_X_0 = df_X_0[['KIDS_TILE', 'KIDS_ID', 'LABEL', 'FOLDER']]
    df_X_1 = df_X_1[['KIDS_TILE', 'KIDS_ID', 'LABEL', 'FOLDER']]

    df_X = pd.concat([df_X_0, df_X_1], ignore_index=True).dropna().reset_index(drop=True)

    df_X_train_tmp, df_X_test, Y_train_tmp, Y_test = train_test_split(df_X, df_X['LABEL'].values, test_size=0.20, random_state=42, stratify=df_X['LABEL'].values)
    df_X_train, df_X_val, Y_train, Y_val = train_test_split(df_X_train_tmp, df_X_train_tmp['LABEL'].values, test_size=0.10, random_state=42, stratify=df_X_train_tmp['LABEL'].values)

    df_train = df_backup_traintest(df_X_train, 'train')
    df_test = df_backup_traintest(df_X_test, 'test')
    df_traintest = pd.concat([df_train, df_test], ignore_index=True)

    utils.folderexist('Retrain/' + folder)

    df_traintest.to_csv(os.path.join(abs_p, 'Retrain', folder, 'df_train_test.csv'), index=False)
    df_X_val.to_csv(os.path.join(abs_p, 'Retrain', folder, 'df_val.csv'), index=False)

    X_train = findlenses.createdataset_from_path(df_X_train['FOLDER'].values, df_X_train['KIDS_ID'].values, df_X_train['KIDS_TILE'].values, return_idx_img=False)
    X_test  = findlenses.createdataset_from_path(df_X_test['FOLDER'].values, df_X_test['KIDS_ID'].values, df_X_test['KIDS_TILE'].values, return_idx_img=False)
    X_val   = findlenses.createdataset_from_path(df_X_val['FOLDER'].values, df_X_val['KIDS_ID'].values, df_X_val['KIDS_TILE'].values, return_idx_img=False)

    Y_train  = np.expand_dims(np.asarray(Y_train).astype(np.int32), axis=1)
    Y_test   = np.expand_dims(np.asarray(Y_test).astype(np.int32), axis=1)
    Y_val    = np.expand_dims(np.asarray(Y_val).astype(np.int32), axis=1)

    return X_train, X_test, X_val, Y_train, Y_test, Y_val, df_X_val


# Function to check lenses on specified tiles using trained models
def check_on_tile(folder_weights, list_tile_idx=settings.tile_not_seen, save_all=False, run_duplicate=False, model_name='Lens_15'):
    """
    Parameters:
    - folder_weights: Folder containing saved weights for trained models
    - list_tile_idx: List of tile indices to check for lenses
    - save_all: Flag to save all predictions to CSV
    - run_duplicate: Flag to run even if the file already exists
    - model_name: Base name for the model

    Output:
    - CSV files with lens predictions and an output text file with used model information
    """
    path_weights = os.path.join('Retrain', folder_weights)
    path_model_used = []

    for tile_idx in list_tile_idx:
        cutclass = gen_cutouts.Cutouts()
        name = sorted(set(cutclass.cats['Tile name']))[tile_idx].rstrip(' ')
        print(f'Checking tile {name}')

        table_tile_test = cutclass.getTableTile(name)
        tab_to_use = gen_cutouts.apply_preprocessing(table_tile_test)  # VERY IMPORTANT STEP

        model_name = model_name + '_' + name

        if save_all:
            model_name = f'ALL_z_{settings.z_min}_{settings.z_max}_' + model_name

        if os.path.exists(os.path.join(path_weights, model_name + '.csv')) and not run_duplicate:
            print('File exists already, skipping it')
            continue

        if 'NO_retrain' in folder_weights:
            print('######### NO weights saved  #########')
            findlenses.findlenses_in_dataset(settings.path_to_save_imgs, tab_to_use['ID'].data, tab_to_use['KIDS_TILE'].data, model=models.lens15(),
                                             model_name=model_name, path_csv=path_weights, channels=['r', 'i', 'g', 'u'], n_img_batch=500, save_all=save_all)
            path_model_used.append(settings.path_weights)
        else:
            print(f'Using weights from {path_weights}')
            findlenses.findlenses_in_dataset(settings.path_to_save_imgs, tab_to_use['ID'].data, tab_to_use['KIDS_TILE'].data,
                                             model=models.lens15(os.path.join(path_weights, 'weights.h5')),
                                             model_name=model_name, path_csv=path_weights,
                                             channels=['r', 'i', 'g', 'u'], n_img_batch=500, save_all=save_all)
            path_model_used.append(path_weights)

        with open(path_weights + "/Output.txt", "w") as text_file:
            text_file.write(" \n".join(str(item) for item in path_model_used))


# Function to create a DataFrame with noisy images
def make_df_with_noise(mu=0, sigma=10**(-15)):
    """
    Parameters:
    - mu: Mean of the Gaussian noise
    - sigma: Standard deviation of the Gaussian noise

    Output:
    - CSV file containing a DataFrame with information about noisy images
    """
    lenses_by_kids = pd.read_csv(settings.catalog_lens_kidscollab)
    df_to_add_noise = pd.read_csv(os.path.join(abs_p, 'Retrain/Rotated/df_rot.csv'))
    df_to_add_noise = pd.concat([df_to_add_noise, lenses_by_kids.dropna()], ignore_index=True).reset_index(drop=True)

    img_tmp = utils.from_fits_to_array(settings.path_lens_foundby_kids, lenses_by_kids['KIDS_ID'][0], lenses_by_kids['KIDS_TILE'][0])
    gaussian = np.random.normal(mu, sigma, (img_tmp.shape[1], img_tmp.shape[2]))

    # create dataset
    gaussian = np.random.normal(0, 10**(-11), (img_tmp.shape[1], img_tmp.shape[2]))
    ID_list = []
    for i in range(len(df_to_add_noise)):
        for b in range(4):
            img_tmp = utils.from_fits_to_array(df_to_add_noise['FOLDER'][i], df_to_add_noise['KIDS_ID'][i], df_to_add_noise['KIDS_TILE'][i], channels=settings.channels[b])
            noisy_image = np.zeros_like(img_tmp)
            noisy_image[:, :] = img_tmp[:, :] + gaussian
            path_tmp = output_path_augmented(settings.channels[b], df_to_add_noise['KIDS_TILE'][i], df_to_add_noise['KIDS_ID'][i], 'NoiseGauss_' + str(mu) + '-' + str(sigma) + '_')
            utils.folderexist('/'.join(path_tmp.split('/')[:-2]))
            utils.folderexist('/'.join(path_tmp.split('/')[:-1]))
            hdu_sci = fits.PrimaryHDU(noisy_image[0, :, :])
            hdu = fits.HDUList([hdu_sci])
            hdu.writeto(path_tmp, overwrite=True)
        ID_list.append('.'.join(path_tmp.split('/')[-1].split('.')[:-1]))

    # create DataFrame with dataset info
    df = pd.DataFrame()
    df['KIDS_TILE'] = df_to_add_noise['KIDS_TILE'].values
    df['KIDS_ID'] = ID_list
    df['LABEL'] = 1
    df['FOLDER'] = path_aug
    df.to_csv(f'Retrain/Rotated+NoiseGauss_mu_{mu}_sigma_{sigma}/df_rot+noisegauss.csv', index=False)


# Function to rotate images
def rot(path_folder_aug='NewRotated', csv_1='df_newrot.csv'):
    """
    Parameters:
    - path_folder_aug: Folder where the dataframe to load is saved
    - csv_1: Name of the dataframe containing the 1s

    Output:
    - Rotated and retrained models with the given dataframes
    """
    lenses_by_kids = pd.read_csv(settings.catalog_lens_kidscollab).dropna()

    # load the 1s
    df_1 = pd.read_csv(os.path.join(abs_p, 'Retrain', path_folder_aug, csv_1))
    df_1 = df_1[['KIDS_TILE', 'KIDS_ID', 'LABEL', 'FOLDER']]
    lenses_by_kids = lenses_by_kids[['KIDS_TILE', 'KIDS_ID', 'LABEL', 'FOLDER']]
    df_1 = pd.concat([df_1, lenses_by_kids], ignore_index=True).dropna().reset_index(drop=True)

    # prepare the 0s
    path_0 = 'Retrain/NO_retrain/'
    cuclass = gen_cutouts.Cutouts()
    list_name = list(sorted(set(cuclass.cats['Tile name'])))

    list_tiles_NO_ret = []
    for i in settings.tile_idx:
        list_tiles_NO_ret.append(path_0 + 'lens_15_' + list_name[i].strip() + '.csv')

    df_0_most_info = pd.DataFrame()

    for d_path in list_tiles_NO_ret:
        df_0 = pd.read_csv(d_path)
        df_0['FOLDER'] = settings.path_to_save_imgs
        df_sorted = df_0[df_0['LABEL'] == 0].sort_values(by=['prob'], ascending=False)[:150][['KIDS_TILE', 'KIDS_ID', 'LABEL', 'FOLDER']]
        df_0_most_info = pd.concat([df_0_most_info, df_sorted]).reset_index(drop=True)

    idx_0 = random.sample(range(0, len(df_0_most_info)), len(df_1))  # same amount of elements like the lenses
    df_0_most_info = df_0_most_info.iloc[idx_0]

    # train
    X_train, X_test, X_val, Y_train, Y_test, Y_val, df_val = train_test_val(df_0_most_info, df_1, folder=path_folder_aug)
    re_train(X_train, Y_train, X_test, Y_test, folder=path_folder_aug)

    # apply model on the tiles
    check_on_tile(path_folder_aug)





if __name__=="__main__":

    #rot()
    # create a process pool that uses all cpus
    #with Pool() as pool:
    #    for t in pool.map(check_on_tile, settings.tile_idx):
    #        pool.map(check_on_tile, t)
    #        print(t)



    #rot(folder_new = 'Rotated_Flipped_Transp', csv_1='df_rot_flip_transp.csv')
    #rot()
    check_on_tile( 'NewRotated')