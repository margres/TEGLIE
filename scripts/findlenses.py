#import tensorflow as tf
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
#import matplotlib.pyplot as plt
#from PIL import Image
import pandas as pd
from .preprocessing import scaling_clipping
from . import settings, gen_cutouts, utils  

def createdataset_from_path(path_dat,
                            kids_id_list,
                            kids_tile_list,
                            channels = settings.channels, 
                            return_idx_img=False, 
                            apply_preproc=True,
                            verbose = True ):

    '''
    Create a dataset from FITS files specified by the given parameters.

    Parameters:
        path_dat (str or list): Folder or list of folders containing the FITS images.
        kids_id_list (list): List of KiDS IDs.
        kids_tile_list (list): List of KiDS tile names.
        channels (list): List of channel indices.
        return_idx_img (bool): If True, return the indices of images found.
        apply_preproc (bool): If True, apply pre-processing to the images.

    Returns:
        np.ndarray: Array containing the dataset.
        list: Indices of images found if return_idx_img is True.
    '''

    X = np.zeros((1,101,101,len(channels)))
    idx_found_tiles = []
    for i in range(len(kids_id_list)):
        try:
            if isinstance(path_dat, str): 
                #if path is a string
                X_tmp = utils.from_fits_to_array(path_dat, kids_id_list[i] , kids_tile_list[i], channels=channels)
                if verbose:
                    print(f'loaded img from folder {path_dat}, ID {kids_id_list[i]}, TILE {kids_tile_list[i]} ')
            else: 
                # if list, iterate
                X_tmp = utils.from_fits_to_array(path_dat[i], kids_id_list[i] , kids_tile_list[i], channels=channels)
                if verbose:
                    print(f'loaded img from folder {path_dat}, ID {kids_id_list[i]}, TILE {kids_tile_list[i]}')
        except FileNotFoundError:
            try:
               # Try to create cutouts if not found
                X_tmp = utils.from_fits_to_array(settings.path_to_save_imgs_alternative, kids_id_list[i] , kids_tile_list[i], channels=channels)
                if verbose:
                    print(f'loaded img from folder {path_dat}, ID {kids_id_list[i]}, TILE {kids_tile_list[i]}')
            except FileNotFoundError:
                cutclass = gen_cutouts.Cutouts()
                try: 
                    #maybe i did not do the cutout
                    gen_cutouts.cutout_by_name_tile(kids_tile_list[i],
                        cutclass.getTableTile(kids_tile_list[i])[cutclass.getTableTile( kids_tile_list[i])['ID']== kids_id_list[i]],
                        channels=channels, apply_preproc=apply_preproc , path_to_save= settings.path_to_save_imgs_alternative)
                    X_tmp = utils.from_fits_to_array(settings.path_to_save_imgs_alternative, kids_id_list[i], kids_tile_list[i], channels=channels )
                except:
                    continue
            except OSError:
                continue
        except OSError:
            continue

        X_tmp = np.expand_dims(np.transpose(X_tmp, axes=(1,2,0)), axis=0 )
        idx_found_tiles.append(i)
        X = np.vstack(( X, X_tmp ))
    X=X[1:]
    X = scaling_clipping(X)
    if return_idx_img:
        return X, idx_found_tiles
    else:
        return X


def populatedict(kids_id, kids_tile, predicted_lenses_prob, model_name):

    '''
    Function to create a dictionary with important information about a lens.

    Args:
    - kids_id (str): ID of the lens.
    - kids_tile (str): Tile information of the lens.
    - predicted_lenses_prob (float): Probability of the lens being present.
    - model_name (str): Name of the model used.
    - ra (str): Right Ascension of the lens.
    - dec (str): Declination of the lens.

    Returns:
    - catalog_candidates: Dictionary containing lens information.
    '''

    catalog_candidates = {
        'KIDS_TILE': kids_tile,
        'KIDS_ID': kids_id,
        'prob': np.round(predicted_lenses_prob, 3),
        'LABEL': None,
        'model': str(model_name)}
    
    return catalog_candidates

def save_dict(catalog_candidates, tile_name, n_model, tile_idx, folder_path):

    '''
    Save a dictionary with lenses' information to a CSV file.

    Args:
    - lensed (dict): Dictionary containing lens information.
    - tile_name (str): Name of the tile.
    - n_model (int): Model number.
    - tile_idx (int): Tile index.
    - folder_path (str): Path to the folder where the CSV will be saved.

    Returns:
    - None
    '''

    df_candidates = pd.DataFrame.from_dict(catalog_candidates)
    df_candidates.to_csv(os.path.join(folder_path, f'{tile_name}_model-{n_model}_tile-{tile_idx}.csv'), index=False)



def findlenses_in_dataset(path_dat,kids_id_list,kids_tile_list, 
                          n_img_batch= 100, model= None,
                        folder_output=None, model_name=None, 
                        channels=settings.channels, return_idx_prob=False, 
                        save_to_csv=True,
                        apply_preproc = True, save_all=False,
                        verbose = True):

    '''
    Main function to find lenses using already available models and weights.
    It saves CSV with a list of found lenses.

    Args:
    - path_dat (str): Path to the images.
    - kids_id_list (list): List of lens IDs.
    - kids_tile_list (list): List of tile information for lenses.
    - n_img_batch (int): Number of images in each batch.
    - model (tf.keras.Model): Pre-trained lens detection model.
    - folder_output (str): Path to save the CSV file.
    - model_name (str): Name of the model.
    - channels (list): List of channels.
    - return_idx_prob (bool): Whether to return indices and probabilities.
    - save_to_csv (bool): Whether to save the results to a CSV file.
    - apply_preproc (bool): Whether to apply preprocessing to images.
    - save_all (bool): Whether to save results for all images in the batch.

    Returns:
    - Tuple or None: Indices and probabilities of detected lenses if `return_idx_prob` is True, otherwise None.
    '''


    ##create batches
    tot_img = len(kids_id_list)

    idx_list = np.arange(0,tot_img,n_img_batch)
    if (tot_img - idx_list[-1])>0:
        idx_list = np.hstack((idx_list,tot_img))

    predicted_lenses_idx ,predicted_lenses_prob = [],[]
    count=0
    count_lens = 0
    utils.folderexist(folder_output)

    path_csv=os.path.join(folder_output,str(model_name.strip()) +f'_z_{str(settings.z_min)}_{str(settings.z_max)}')
    predicted_lenses_idx ,predicted_lenses_prob = [],[]

    for i in range(len(idx_list)-1):

        #hard coding batches
        X, _ = createdataset_from_path(path_dat,
                                                   kids_id_list[idx_list[i]:idx_list[i+1]],
                                                   kids_tile_list[idx_list[i]:idx_list[i+1]],
                                                   channels=channels, 
                                                   return_idx_img=True, 
                                                   apply_preproc=apply_preproc,
                                                   verbose=verbose)

        if X.size>1:

            count+=len(X)
            Y_pred = model.predict(X).ravel()

            predicted_lenses_idx_tmp = np.where(Y_pred>settings.threshold)[0]
            count_lens += len(predicted_lenses_idx_tmp)
            if len(predicted_lenses_idx_tmp) >0:
                print('\n# lens found {}'.format(len(predicted_lenses_idx_tmp)))

            if save_all:
                predicted_lenses_idx_tmp = np.arange(0,len(Y_pred))

            predicted_lenses_prob_tmp = Y_pred[predicted_lenses_idx_tmp]

            #order matters
            if idx_list[i]!=0:
                predicted_lenses_idx_tmp += idx_list[i]

            predicted_lenses_idx.extend(predicted_lenses_idx_tmp)
            predicted_lenses_prob.extend(Y_pred.ravel())

        else:
            pass 

        if save_to_csv and len(predicted_lenses_prob_tmp)>0: 

            lensed = populatedict(kids_id_list[predicted_lenses_idx_tmp],kids_tile_list[predicted_lenses_idx_tmp],predicted_lenses_prob_tmp,model_name)
            df_lensed = pd.DataFrame.from_dict(lensed)

            if os.path.isfile(path_csv+'.csv')  and  (len(predicted_lenses_idx) > len(predicted_lenses_idx_tmp) ): 

                #not the first batch
                df_in_memory = pd.read_csv(path_csv+'.csv')
                df_tosave = pd.concat([df_in_memory, df_lensed]).reset_index(drop=True)

            elif os.path.isfile(path_csv+'.csv')  and (len(predicted_lenses_idx) == len(predicted_lenses_idx_tmp) ): 

                #if first batch and results already in memory
                print('Results with the same path have been found, saving those as "ver_2"')
                print(f'\nResults are saved as {path_csv}.csv')
                path_csv=path_csv+'_ver2'
                df_tosave = df_lensed

            else:
                 #if first batch and results not in memory
                print(f'\nResults are saved as {path_csv}.csv')
                df_tosave = df_lensed

            df_tosave.to_csv(path_csv+'.csv', index=False, mode='w+')

    print('\nFound {}/{} (sgl candidate)/(tot checked) \n'.format(count_lens, count))
    if return_idx_prob:
        return predicted_lenses_idx, predicted_lenses_prob

if __name__ == '__main__':
    findlenses_in_dataset()
