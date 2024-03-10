import pandas as pd
import numpy as np
import os
import  math
from . import gen_cutouts, settings, utils, make_rgb
import sys
from IPython.display import clear_output


def put_already_labeled(path_to_label, path_labeled, force_label=False, label_nan=False):
    """
    Updates labels in a new CSV file using information from an already labeled CSV file.
    
    Parameters:
    - path_new: Path to the new CSV file.
    - path_labeled: Path to the already labeled CSV file.
    - force_label: If set to 'first' or 'second', enforces the label from the specified file in case of conflicts.
    - label_nan: If True, allows labeling instances with NaN labels based on probabilities.

    This function reads two CSV files, compares labels for common elements, and updates the new CSV file accordingly.
    """
    df_to_label = pd.read_csv(path_to_label)
    df_to_label.dropna(axis=0, how='all', inplace=True)
    df_labeled = pd.read_csv(path_labeled)
    df_labeled.dropna(axis=0, how='all', inplace=True)

    labels = ['0', '1', '2', 's', '3']

    common_elem_df1_df2 = list(set(df_to_label['KIDS_ID']) & set(df_labeled['KIDS_ID']))
    print('df1 len {} df2 len {} common {}'.format(len(df_to_label), len(df_labeled), len(common_elem_df1_df2)))

    for ID in common_elem_df1_df2:
        lab_df2 = df_labeled['LABEL'][df_labeled['KIDS_ID'] == ID].values[0]
        lab_df1 = df_to_label['LABEL'][df_to_label['KIDS_ID'] == ID].values[0]

        try:
            lab_df1 = float(lab_df1)
            lab_df2 = float(lab_df2)
        except:
            print(ID)
            print(f'first {path_to_label}')
            print(f'second {path_labeled}')

        if math.isnan(lab_df1) and not math.isnan(lab_df2):
            df_to_label['LABEL'][df_to_label['KIDS_ID'] == ID] = lab_df2
            print('gave val')

        elif math.isnan(lab_df1) and math.isnan(lab_df2):
            if (df_labeled['prob'][df_labeled['KIDS_ID'] == ID].values[0] > settings.threshold or
                    df_to_label['prob'][df_to_label['KIDS_ID'] == ID].values[0] > settings.threshold) and label_nan:
                try:
                    img = utils.from_fits_to_array(settings.path_to_save_imgs, ID, df_to_label['KIDS_TILE'][0])
                    utils.plot_oneimg_n_channels(img)
                    make_rgb.make_rgb_one_image(img, display_plot=True)
                    new_label = input(f'give label for 1 is {lab_df1}, for 2 {lab_df2}')
                    if new_label == 's':
                        sys.exit()
                    df_to_label['LABEL'][df_to_label['KIDS_ID'] == ID] = new_label
                    df_labeled['LABEL'][df_labeled['KIDS_ID'] == ID] = new_label
                    df_labeled.reset_index(drop=True).to_csv(path_labeled, index=False)
                    df_to_label.reset_index(drop=True).to_csv(path_to_labelt, index=False)
                    clear_output()
                except FileNotFoundError:
                    pass
            else:
                pass
        else:
            if lab_df1 != lab_df2:
                if force_label == 'first':
                    df_labeled['LABEL'][df_labeled['KIDS_ID'] == ID] = lab_df1
                elif force_label == 'second':
                    df_to_label['LABEL'][df_to_label['KIDS_ID'] == ID] = lab_df2
                else:
                    try:
                        img = utils.from_fits_to_array(settings.path_to_save_imgs, ID, df_to_label['KIDS_TILE'][0])
                        utils.plot_oneimg_n_channels(img)
                        make_rgb.make_rgb_one_image(img, display_plot=True)
                        new_label = None
                        while new_label not in labels:  # continue asking
                            new_label = input(f'diff label, give label for 1 is {lab_df1}, for 2 {lab_df2}')

                        df_to_label['LABEL'][df_to_label['KIDS_ID'] == ID] = new_label
                        df_labeled['LABEL'][df_labeled['KIDS_ID'] == ID] = new_label
                        df_labeled.reset_index(drop=True).to_csv(path_labeled, index=False)
                        clear_output()

                        if new_label == 's':
                            sys.exit()
                    except:
                        pass
            else:
                pass

    df_labeled.reset_index(drop=True).to_csv(path_labeled, index=False)
    df_to_label.reset_index(drop=True).to_csv(path_to_label, index=False)


def plot_and_label(df, iterate_over, path_csv=None, 
        path_imgs=settings.path_to_save_imgs, channels=settings.channels, check_labeled=False, gen_cutout=False):

    lenses_by_kids = pd.read_csv(settings.catalog_lens_kidscollab)
    df4=pd.merge(df['KIDS_ID'],lenses_by_kids['KIDS_ID'], how='inner')


    #if there is not the column add it
    if 'LABEL' not in df.columns:
        df['LABEL'] = None
    #I need an interactive program to save the labels of the test set - labels 0 , 1 ,2 (maybe)
    labels = ['0','1','2', 's', '3', 'b']

    count=0
    idx = iterate_over[count]

    while idx < iterate_over[-1]:
       
        print('{}/{}'.format(count,len(iterate_over )))
        print( 'TILE {} ---- ID {}'.format(df['KIDS_TILE'][idx] ,df['KIDS_ID'][idx]))
        try:
            img_tmp = utils.from_fits_to_array(path_imgs, df['KIDS_ID'][idx], df['KIDS_TILE'][idx])
            path_imgs_=path_imgs
        except FileNotFoundError:
            try: 
                #check if it's in the other HD
                img_tmp = utils.from_fits_to_array('/home/grespanm/mnt/HD_MG/KiDS_cutout', df['KIDS_ID'][idx], df['KIDS_TILE'][idx])
                path_imgs_ = '/home/grespanm/mnt/HD_MG/KiDS_cutout'
                print('in HD_MG')
            except FileNotFoundError:
                if gen_cutout:
                    try:
                        print(' make cutout ')
                        path_imgs_ = settings.path_to_save_imgs 
                        # '/home/grespanm/mnt/HD_MG/KiDS_cutout'
                        cutclass = gen_cutouts.Cutouts()
                        a,b = df['KIDS_ID'][idx], df['KIDS_TILE'][idx]
                        gen_cutouts.cutout_by_name_tile(b, cutclass.getTableTile(b)[cutclass.getTableTile(b)['ID']==a], apply_preproc=True, path_to_save=path_imgs_)
                        img_tmp = utils.from_fits_to_array(path_imgs, df['KIDS_ID'][idx], df['KIDS_TILE'][idx])
                    except Exception as e:
                        count +=1
                        idx = iterate_over[count]
                        print(f'Cutout not available - Error: {e}')
                        continue
                else: 
                    count +=1
                    idx = iterate_over[count]
                    continue
            except Exception as e:
                count +=1
                idx = iterate_over[count]
                print(f'Cutout not available - Error: {e}')
                continue
        except Exception as e:
            #print(e)
            count +=1
            idx = iterate_over[count]
            print(f'Cutout not available - Error: {e}')
            continue

        utils.plot_oneimg_n_channels(img_tmp, channels)
        make_rgb.make_rgb_one_image(img_name=df['KIDS_ID'][idx],
                                    tile_name=df['KIDS_TILE'][idx], display_plot=True, folder_cutouts=path_imgs_ )
        label_tmp = None
        if df['KIDS_ID'][idx] in df4['KIDS_ID'].tolist():
            print('####### LENS by KiDS #######')
        if check_labeled==False:
            while label_tmp  not in labels: #continue asking
                label_tmp = str(input("Please enter the label 0-1-2-3-s-b: "))
                if label_tmp == 's':
                    sys.exit()
                elif label_tmp == 'b':
                    if count==0:
                        iterate_over = np.insert(iterate_over, 0,iterate_over[count]-1)
                    else:
                        count-=2
                    print(iterate_over)
        else: 
            print('previous label: ',  df['LABEL'].loc[idx])
            while label_tmp  not in labels and label_tmp != '': #continue asking
                label_tmp = str(input("Please enter the label 0-1-2-3-s-b: "))
                if label_tmp== '':
                    pass
                elif label_tmp == 's':
                    sys.exit()
        if  label_tmp != 'b':
            df['LABEL'].loc[idx] = label_tmp

        clear_output()
        if path_csv is not None:
            df.reset_index(drop=True).to_csv(path_csv, index=False)
        count +=1
        idx = iterate_over[count]
        print(idx)

    return df


def labeling(path_csv, path_imgs=settings.path_to_save_imgs,channels=settings.channels, 
              only_candidates=True, check_labeled=False, gen_cutout=False):

    ##### put labels to the object labeled in other folders ##############
    list_folders = settings.folders_retrain
    list_folders = [os.path.join('Retrain',l) for l in list_folders]

    if  'BACKUP' not in path_csv:
        pass
    else:
        print('you are labeling the backup')
        return

    for folder in list_folders:
        try:
            put_already_labeled(path_csv, os.path.join(folder,path_csv.split('/')[-1]))
        except FileNotFoundError:
            pass

    ############# start the labeling ############
    df = pd.read_csv(path_csv)
    df.drop(df.columns[np.where(['Unnamed' in col for col in df.columns.values])[0]], axis=1, inplace= True)
    #df = df.sort_values(by=['prob'],ascending=False)
    df = df.drop_duplicates(subset= ['KIDS_ID']).reset_index(drop=True)
    df.to_csv(path_csv, index=False)

    if only_candidates==True and check_labeled==False:
        #iterate_over = np.intersect1d(np.where(df['LABEL'].isna()==True)[0], df[df['prob']>settings.threshold].index.values)
        iterate_over = df.loc[~df['LABEL'].isin([0, 1, 2, 3]) & (df['prob'] > settings.threshold)].index.values

    elif only_candidates==True and check_labeled==False:
        iterate_over = np.intersect1d(df[df['prob']>settings.threshold].index.values)
    else:
        iterate_over = np.where(df['LABEL'].isna()==True)[0]

    if len(iterate_over)>0:
        df = plot_and_label(df, iterate_over,path_csv=path_csv,  
                            path_imgs=settings.path_to_save_imgs,channels=settings.channels, check_labeled=check_labeled, gen_cutout=gen_cutout)


def cut_from_coord(tile_name,ra,dec, path_to_save=settings.path_to_save_imgs,channels =settings.channels, ):
    #print(tab)
    for f in channels:
        #prepare name to save it
        if '_DR4.0' in tile_name:
            pass
        else:
            tile_name = tile_name.replace('KIDS', 'KiDS_DR4.0')
        tile_name = tile_name.strip()

        new_path = os.path.join(path_to_save,f+'_band/' )

        if not os.path.exists(new_path):
            os.mkdir(new_path)

        sci_suff = '_'+f+'_sci.fits'

        new_path = os.path.join(new_path,utils.change_specialcharacters(tile_name))

        if not os.path.exists(new_path):
            os.mkdir(new_path)

        try:
            sci_tile = fits.open(os.path.join(settings.path_kids_data, 'ugri_coadds',tile_name+sci_suff))
        except FileNotFoundError:
            print('tile {} not available'.format(tile_name))
            return

        sci_wcs = WCS(sci_tile[0].header)
        sci_wcs.sip = None

        ident = 'ra_{}_dec_{}'.format(ra,dec) #name to save it

        pos = SkyCoord(ra*u.deg, dec*u.deg)
        path_tmp = os.path.join(new_path,str(ident)+'.fits')
        #print(path_tmp)
        if os.path.isfile(path_tmp): #if fiel exist do not create it agaion
            continue
        try:
        #print(sci_tile[0].data)
        #print(np.isnan(sci_tile[0].data).any())
            cutout_sci = Cutout2D(sci_tile[0].data, pos, 101, wcs=sci_wcs)
        except:
            print('error with the cutout')
            print(ra,dec,tile_name)
            #traceback.print_exc()
            #raise ValueError
            continue
        #return ValueError('Error with the cutout')
        hdu_sci = fits.PrimaryHDU(cutout_sci.data, header=cutout_sci.wcs.to_header())
        hdu = fits.HDUList([hdu_sci])
        hdu.writeto(path_tmp,overwrite=True)

        