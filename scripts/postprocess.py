from teglie import gen_cutouts, settings, utils, make_rgb
import pandas as pd
import numpy as np
import sys
import os 
from itertools import combinations


p='/home/grespanm/github/FiLeK/'


def check_elements(value_counts, label):
    """
    Check the number of occurrences of a label in a pandas Series.

    Parameters:
    - value_counts: Pandas Series containing value counts.
    - label: Label to check.

    Output:
    - Number of occurrences of the label.
    """
    try:
        amount = value_counts[label]
        return amount
    except KeyError:
        return 0

def put_values(df, n_elem, model_1_tmp, model_2_tmp, n_diag_1, n_diag_2):
    """
    Update values in a DataFrame based on element counts.

    Parameters:
    - df: Pandas DataFrame.
    - n_elem: Number of elements.
    - model_1_tmp: Model 1 label.
    - model_2_tmp: Model 2 label.
    - n_diag_1: Number for diagonal 1.
    - n_diag_2: Number for diagonal 2.

    Output:
    - Updates the DataFrame in-place.
    """
    df.iloc[df[df['Model'] == model_2_tmp].index.values[0], df.columns.get_loc(model_1_tmp)] = n_elem
    df.iloc[df[df['Model'] == model_1_tmp].index.values[0], df.columns.get_loc(model_2_tmp)] = n_elem

    ##diagonal
    df.iloc[df[df['Model'] == model_1_tmp].index.values[0], df.columns.get_loc(model_1_tmp)] = n_diag_1
    df.iloc[df[df['Model'] == model_2_tmp].index.values[0], df.columns.get_loc(model_2_tmp)] = n_diag_2

def check_duplicates(df):
    """
    Check for duplicates in a DataFrame based on RA and DEC.

    Parameters:
    - df: Pandas DataFrame.

    Output:
    - List of indices to drop.
    """
    df = df.reset_index(drop=True)
    to_drop = []
    for i in range(len(df)):
        if i in to_drop:
            continue
        check_dec = np.where((abs(df['DEC'][i] - df['DEC']) < 0.01) & (abs(df['DEC'][i] - df['DEC']) != 0))[0]
        check_ra = np.where((abs(df['RA'][i] - df['RA']) < 0.01) & (abs(df['RA'][i] - df['RA']) != 0))[0]
        inters = np.intersect1d(check_ra, check_dec)
        if len(inters) > 0:
            to_drop.append(inters[0])
    return to_drop

def make_list_all_SGL(grade=1, plot=False):
    """
    Create a list of strong gravitational lensing candidates.

    Parameters:
    - grade: Grading for candidates.
    - plot: Flag to display plots.

    Output:
    - Saves a CSV file with the list of candidates.
    - If plot is True, displays plots for each candidate.
    """
    list_folders = settings.folders_retrain
    list_folders = [os.path.join(p, 'Retrain', l) for l in list_folders]

    df_ones = pd.DataFrame()
    for f in list_folders:
        df_ones_tmp = pd.read_csv(os.path.join(f, 'SL_candidates_grade_{}.csv'.format(int(grade)))).reset_index(drop=True)
        df_ones = pd.concat([df_ones, df_ones_tmp], ignore_index=True, keys=[df_ones_tmp.columns[0], df_ones_tmp.columns[1]]).reset_index(drop=True).drop_duplicates(subset=['KIDS_ID'])
    df_ones = df_ones.drop_duplicates(subset=['KIDS_ID']).reset_index(drop=True)
    idx_double = check_duplicates(df_ones)
    df_final = df_ones.drop(idx_double).reset_index(drop=True)
    df_final.to_csv(os.path.join(p, 'All_SL_candidates_grade_{}.csv'.format(int(grade))))
    if plot:
        lenses_by_kids = pd.read_csv(os.path.join(p, 'lens_catalog.csv')).dropna().reset_index(drop=True)
        for tile_name, ID in zip(df_final['KIDS_TILE'], df_final['KIDS_ID']):
            print(ID, utils.getinfo_from_catalog_tile(ID, tile_name))
            img_tmp = utils.from_fits_to_array(settings.path_to_save_imgs, ID, tile_name)
            utils.plot_oneimg_n_channels(img_tmp)
            make_rgb.make_rgb_one_image(img_tmp, display_plot=True)
        vals = set(df_ones['KIDS_ID']).intersection(lenses_by_kids['KIDS_ID'])
        print('In common with KiDS', len(vals))




def change_label_all_folders_in_retrain(new_label, ID, tile_name):
    """
    Change the label of a specific candidate in a folder.

    Parameters:
    - new_label: New label to assign.
    - ID: KIDS_ID of the candidate.
    - tile_name: Tile name of the candidate.

    Output:
    - Modifies CSV files in the specified folder.
    """

    list_folders = settings.folders_retrain
    list_folders = [os.path.join(p, 'Retrain', l) for l in list_folders]

    for folder in list_folders:
        df_tmp = pd.read_csv(os.path.join(folder, 'Candidates_lens_15_' + tile_name + '.csv'))
        if any(df_tmp['KIDS_ID'] == ID) == True:
            old_label = df_tmp[df_tmp['KIDS_ID'] == str(ID)]['LABEL'].values[0]
            if old_label == new_label:
                print('Already with {} label in folder {}'.format(old_label, folder))
                continue
            print('Lens found in folder {} with label {}'.format(folder, old_label))
            df_tmp.loc[df_tmp['KIDS_ID'] == ID, 'LABEL'] = new_label
            df_tmp.reset_index(drop=True).to_csv(os.path.join(folder, 'Candidates_lens_15_' + tile_name + '.csv'), index=False)

            if old_label != 0:
                df_tmp_SL = pd.read_csv(os.path.join(folder, 'SL_candidates_grade_{}.csv'.format(int(old_label))))
                row_to_change = df_tmp_SL[df_tmp_SL['KIDS_ID'] == ID]
                if len(np.where(df_tmp_SL['KIDS_ID'] == ID)[0]) == 0:
                    raise ValueError('Something wrong')
                df_tmp_SL = df_tmp_SL.drop(np.where(df_tmp_SL['KIDS_ID'] == ID)[0]).reset_index(drop=True)
                df_tmp_SL.to_csv(os.path.join(folder, 'SL_candidates_grade_{}.csv'.format(int(old_label))), index=False)

                if new_label.strip() != str(0):
                    df_tmp_SL = pd.read_csv(os.path.join(folder, 'SL_candidates_grade_{}.csv'.format(int(new_label))))
                    df_tmp_SL = df_tmp_SL.append(row_to_change)
                    df_tmp_SL.to_csv(os.path.join(folder, 'SL_candidates_grade_{}.csv'.format(int(new_label))), index=False)

            if new_label != 0:
                make_list_all_SGL(grade=new_label)
            if old_label != 0:
                make_list_all_SGL(grade=old_label)


def create_catalog_per_folder(grade, list_tiles, list_folders=settings.folders_retrain):
    """
    Create a catalog for a specific grade in each folder.

    Parameters:
    - grade: Grading for candidates.
    - list_tiles: List of tiles.
    - list_folders: List of folders to process.

    Output:
    - Saves a CSV file with the catalog for each folder.
    """
    for f in list_folders:
        df_ones = pd.DataFrame()
        for tile in list_tiles:
            path = os.path.join(p, 'Retrain', f, tile)
            df = pd.read_csv(path)
            if len(df[df['LABEL'] == grade].values) > 0:
                df_ones = pd.concat([df_ones, df[df['LABEL'] == grade]], ignore_index=True, keys=[df.columns[0], df.columns[1]]).reset_index(drop=True)
        df_with_info = pd.DataFrame(columns=['KIDS_TILE', 'KIDS_ID', 'RA', 'DEC', 'z', 'mag'])
        info_list = []
        if len(df_ones) > 0:
            for IDtile, IDimg in df_ones.loc[:, ['KIDS_TILE', 'KIDS_ID']].to_numpy():
                info_list.append(utils.getinfo_from_catalog_tile(IDimg, IDtile))
            df_with_info = pd.DataFrame(info_list, columns=['KIDS_TILE', 'KIDS_ID', 'RA', 'DEC', 'z', 'mag'])
            df_with_info.to_csv(os.path.join(p, 'Retrain', f, 'SL_candidates_grade_{}.csv'.format(grade)), index=False)


def add_info():
    """
    Add information to existing CSV files.
    """
    list_tiles = []
    p = '/home/grespanm/github/FiLeK/'
    cuclass = gen_cutouts.Cutouts()
    list_name = list(sorted(set(cuclass.cats['Tile name'])))
    for i in settings.tile_not_seen:
        list_tiles.append('lens_15_' + list_name[i].strip() + '.csv')

    list_folders = settings.folders_retrain
    list_folders = [os.path.join(p, 'Retrain', l) for l in list_folders]
    list_names_10_tiles = [list_name[i].strip() for i in settings.tile_not_seen]

    for f in list_folders:
        for tile in list_tiles:
            pp = pd.read_csv(os.path.join(f, tile))
            info_list = []
            for IDtile, IDimg in pp.loc[:, ['KIDS_TILE', 'KIDS_ID']].to_numpy():
                info_list.append(utils.getinfo_from_catalog_tile(IDimg, IDtile))
                df_with_info = pd.DataFrame(info_list, columns=['KIDS_TILE', 'KIDS_ID', 'RA', 'DEC', 'z', 'mag'])
                df_with_info = pd.concat([df_with_info, pp[['LABEL', 'prob']]], axis=1)
                df_with_info.to_csv(os.path.join(f, 'Candidates_' + tile), index=False)
