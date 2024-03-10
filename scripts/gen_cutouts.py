import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS
from . import settings


def change_specialcharacters(string):
    return string.replace('KIDS', 'KiDS_DR4.0').replace('.', 'p').replace('-', 'm').rstrip(' ')

#function to select correctly flagged galaxies
def unmasked(tbl, msk):
    binary = []
    for i in range(0, len(tbl)):
        binary.append(np.binary_repr(np.array(tbl['MASK'][i]), 16))
    tbl['MASK_BINARY'] = binary

    good = []
    for j in range(0, len(tbl)):
        if tbl['MASK_BINARY'][j][msk[0]] == '0':
            good.append(j)
    #print(0, len(good))
    if len(msk) > 1:
        for i in range(1, len(msk)):
            good_i = []
            for j in range(0, len(tbl)):
                if tbl['MASK_BINARY'][j][msk[i]] == '0':
                    good_i.append(j)
            #print(i, len(good_i))
            good = np.intersect1d(good, good_i)
    return tbl[good]

def apply_preprocessing(table, z_min = settings.z_min , z_max = settings.z_max, preproc = True ):
        
        try:
            table = Table.from_pandas(table)
        except:
            pass
        try:
           #Filter on binary flags
            table = unmasked(table, [14]) #<--- PASS THE IDEX OF THE FLAGS YOU WANT TO USE IN THE LIST HERE
        except:
            pass

        #Cut by most likely redshift
        z_b_low = np.where(table['Z_B'] >= z_min )[0]
        z_b_hig = np.where(table['Z_B'] <= z_max )[0]
        z_b_bin = np.intersect1d(z_b_low, z_b_hig)

        #Cut by peak of the PDF redshift
        z_ml_low = np.where(table['Z_ML'] >= z_min )[0]
        z_ml_hig = np.where(table['Z_ML'] <= z_max)[0]
        z_ml_bin = np.intersect1d(z_ml_low, z_ml_hig)

        z_ml_b_low = np.where(np.greater(table['Z_ML'], table['Z_B_MIN']))[0]
        z_ml_b_hig = np.where(np.less(table['Z_ML'], table['Z_B_MAX']))[0]
        z_ml_b_bin = np.intersect1d(z_ml_b_low, z_ml_b_hig)

        z_bin = np.intersect1d(z_b_bin, z_ml_bin)
        z_bin = np.intersect1d(z_bin, z_ml_b_bin)


        #************** from Petrillo 2019 *****************
        if preproc:
            #preselectioon of good candidates - no glitches etc
            list_good_flag = np.where(table['Flag'] <4)[0]
            list_good_imaflags = np.where(table['IMAFLAGS_ISO']==0)[0]
            #list_good_imaflags = np.where(table['SG2DPHOT']==0)[0]
            good_cand = np.intersect1d(list_good_flag, list_good_imaflags)
            good_cand = np.intersect1d(z_bin, good_cand)
        else:
            good_cand = z_bin

 
        if len(good_cand)<1:
            print('no candidates')
        else:
            return table[good_cand]
        
def save_cutout(tile_name, pos, ident, path_to_save, sci_data, wcs, mode='strict'):
    '''
    Save the cutout to a file.

    Args:
    - tile_name (str): Name of the tile.
    - pos (SkyCoord): Sky coordinates of the cutout.
    - ident (str): Identifier for the cutout.
    - path_to_save (str): Path to save the cutout.
    - sci_data (numpy.ndarray): Scientific data array.
    - wcs (astropy.wcs.WCS): World Coordinate System.
    - mode (str): Mode for creating the cutout ('strict' or 'partial').

    Returns:
    - None
    '''
    new_path = os.path.join(path_to_save, f'{ident}.fits')

    if os.path.isfile(new_path):  # If file exists, do not create it again
        return

    try:
        cutout_sci = Cutout2D(sci_data, pos, 101, wcs=wcs, mode=mode)
    except:
        print('Error with the cutout')
        return

    hdu_sci = fits.PrimaryHDU(cutout_sci.data, header=cutout_sci.wcs.to_header())
    hdu = fits.HDUList([hdu_sci])
    hdu.writeto(new_path, overwrite=True)

def cutout_by_name_tile(tile_name, table_tile, path_to_save=settings.path_to_save_imgs,
                        channels=settings.channels, apply_preproc=True):
    '''
    Parameters
    -------------
    '''
    if apply_preproc:
        table_tile_tmp = apply_preprocessing(table_tile)
        if np.shape(table_tile_tmp)[0] < 1:
            print('with preprocessing no elements')
            return tile_name, table_tile['ID'].tolist()[0]
        else:
            table_tile = table_tile_tmp

    if len(table_tile['KIDS_TILE'].data) < 2:
        table_tile = [table_tile]

    for tab in table_tile:  # iterate per row
        for f in channels:
            if '_DR4.0' not in tile_name:
                tile_name = tile_name.replace('KIDS', 'KiDS_DR4.0')
            tile_name = tile_name.strip()

            new_path = os.path.join(path_to_save, f'{f}_band/')

            if not os.path.exists(new_path):
                os.mkdir(new_path)

            sci_suff = f'_{f}_sci.fits'
            new_path = os.path.join(new_path, change_specialcharacters(tile_name))

            if not os.path.exists(new_path):
                os.mkdir(new_path)

            try:
                sci_tile = fits.open(os.path.join(settings.path_kids_data, 'ugri_coadds', tile_name + sci_suff))
            except FileNotFoundError:
                print('tile {} not available'.format(tile_name))
                return

            sci_wcs = WCS(sci_tile[0].header)
            sci_wcs.sip = None
            RA = tab['RAJ2000']
            DEC = np.asarray(tab['DECJ2000'], dtype=float)
            ident = tab['ID'].tolist()[0]
            if ident == 'K':
                ident = tab['ID'].tolist()
                if ident == 'K':
                    raise ValueError('Wrong name of the file!')

            pos = SkyCoord(RA * u.deg, DEC * u.deg)
            save_cutout(tile_name, pos, ident, new_path, sci_tile[0].data, sci_wcs)

def cut_from_coord(tile_name, ra, dec, path_to_save=settings.path_to_save_imgs, channels=settings.channels):
    for f in channels:
        if '_DR4.0' not in tile_name:
            tile_name = tile_name.replace('KIDS', 'KiDS_DR4.0')
        tile_name = tile_name.strip()

        new_path = os.path.join(path_to_save, f'{f}_band/')

        if not os.path.exists(new_path):
            os.mkdir(new_path)

        sci_suff = f'_{f}_sci.fits'

        new_path = os.path.join(new_path, change_specialcharacters(tile_name))

        if not os.path.exists(new_path):
            os.mkdir(new_path)

        try:
            sci_tile = fits.open(os.path.join(settings.path_kids_data, 'ugri_coadds', tile_name + sci_suff))
        except FileNotFoundError:
            print('tile {} not available'.format(tile_name))
            return

        sci_wcs = WCS(sci_tile[0].header)
        sci_wcs.sip = None

        ident = f'ra_{ra}_dec_{dec}'
        pos = SkyCoord(ra * u.deg, dec * u.deg)
        save_cutout(tile_name, pos, ident, new_path, sci_tile[0].data, sci_wcs)

        
class Cutouts():

    def __init__(self):

        #Setup the paths and data files
        self.pp_mnt =  settings.path_kids_data
        self.data = fits.getdata(os.path.join(self.pp_mnt,'KiDS_DR4_observations_table.fits'), 1)
        self.cats = Table(self.data)

        self.path_imgs=settings.path_to_save_imgs
        if not os.path.exists(self.path_imgs):
            os.mkdir(self.path_imgs)

        #Path to the science images
        self.sci_path = self.pp_mnt+ '/ugri_coadds/'
        #Path to the catalogues
        self.path = self.pp_mnt+'/multi-band_catalogue/'
        self.extn = '_ugriZYJHKs_cat.fits'

        #Redshift limits
        self.z_min = settings.z_min
        self.z_max = settings.z_max


    def getTableTile(self,tile_name, returntilename=False):


        #Open the tile catalogue
        tile_name_tmp = tile_name.replace('KIDS_', 'KiDS_DR4.0_').rstrip(' ')

        try:
            #print(self.path+tile_name_tmp+self.extn)
            data = fits.getdata(self.path+tile_name_tmp+self.extn, 0)

            #create table
            table = Table(data)
            if returntilename:
                return table,tile_name_tmp
            else:
                return table

        except FileNotFoundError :

            #Some catalogues were fixed in DR4.1, so if DR4.0 does not exist, try DR4.1
            tile_name_tmp = tile_name.replace('KIDS_', 'KiDS_DR4.1_').rstrip(' ')

            try:
                data = fits.getdata(self.path+tile_name_tmp+self.extn, 0)
                #create table
                table = Table(data)
                if returntilename:
                    return table,tile_name_tmp
                else:
                    return table
            except FileNotFoundError:
                raise ValueError('File not found!')
