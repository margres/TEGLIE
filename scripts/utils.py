import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import os
import sys
import glob
import matplotlib.pyplot as plt
#there will be loads of warnings that I don't care about, so ignore them
import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings("ignore", category=FITSFixedWarning)
from . import settings
#sys.path.insert(0,'../..')
from . import cutclass

def index_containing_substring(the_list, substring):
    """
    Return the index of the first occurrence of a substring in a list.

    Parameters:
    - the_list (list): List to search in.
    - substring (str): Substring to search for.

    Returns:
    - int: Index of the first occurrence of the substring.

    Raises:
    - ValueError: If the substring is not found in the list.
    """
    for i, s in enumerate(the_list):
        if substring in s:
            return i  # index in the list given
    raise ValueError('Substring not found!')

def change_specialcharacters(string):
    """
    Replace special characters in a string.

    Parameters:
    - string (str): Input string.

    Returns:
    - str: String with replaced special characters.
    """
    return string.replace('KIDS', 'KiDS_DR4.0').replace('.', 'p').replace('-', 'm').rstrip(' ')

def get_idx_from_tilename(tilename):
    """
    Get the index of a tile from its name.

    Parameters:
    - tilename (str): Name of the tile.

    Returns:
    - int: Index of the tile.
    """
    tilename = tilename.strip()
    cats_only_one = cutclass.cats[cutclass.cats['Filter'] == 'r']
    return list(map(str.strip, cats_only_one['Tile name'].data)).index(tilename)

def get_path(tile_idx, f='r'):
    """
    Get the path of all FITS files in one tile for a given filter.

    Parameters:
    - tile_idx (int): Index of the tile.
    - f (str): Filter.

    Returns:
    - list: List of paths to FITS files.
    """
    tile_list = sorted(set(cutclass.cats['Tile name']))
    tile_name = tile_list[tile_idx]
    tile_fldr = change_specialcharacters(tile_name)
    tile_path = sorted(glob.glob(os.path.join(settings.path_to_save_imgs, f + '_band/*')))
    tile_idx_saved = index_containing_substring(tile_path, tile_fldr)
    img_path = sorted(glob.glob(tile_path[tile_idx_saved] + '/*'))
    return img_path

def load_fits(image_file):
    """
    Load FITS file data.

    Parameters:
    - image_file (str): Path to the FITS file.

    Returns:
    - HDUList: FITS file data.
    """
    return fits.open(image_file)[0]

def get_tiles_ra_dec_range(df, dec_low, dec_high, ra_low, ra_high, dec_col_name='DEC', ra_col_name='RA'):
    """
    Get indices of tiles within a specified RA and DEC range.

    Parameters:
    - df: DataFrame or Table with RA and DEC columns.
    - dec_low, dec_high, ra_low, ra_high: Range limits.
    - dec_col_name, ra_col_name: Column names for DEC and RA.

    Returns:
    - ndarray: Array of indices for tiles within the specified range.
    """
    try:
        # pandas
        return np.intersect1d(np.where((dec_low < df[dec_col_name].values) & (df[dec_col_name].values < dec_high)),
                              np.where((ra_low < df[ra_col_name].values) & (df[ra_col_name].values < ra_high))[0])
    except:
        # astropy table
        return np.intersect1d(np.where((dec_low < df[dec_col_name].data) & (df[dec_col_name].data < dec_high)),
                              np.where((ra_low < df[ra_col_name].data) & (df[ra_col_name].data < ra_high))[0])

def get_tile_idx(IDtile):
    """
    Get the index of a tile by its ID.

    Parameters:
    - IDtile (str): Tile ID.

    Returns:
    - int: Index of the tile.
    """
    tile_list = sorted(set(cutclass.cats['Tile name']))
    return index_containing_substring(tile_list, IDtile)

def get_rowtab_by_ID(tile_name, tile_ID):
    """
    Get a row from a table based on tile name and tile ID.

    Parameters:
    - tile_name (str): Name of the tile.
    - tile_ID: ID of the tile.

    Returns:
    - DataFrame or Table: Row corresponding to the specified tile name and ID.
    """
    return cutclass.getTableTile(tile_name)[cutclass.getTableTile(tile_name)['ID'] == tile_ID]

def get_img(fil=settings.channels, tile_idx=None, cutout_idx=None, info=False):
    """
    Load image data from specified tile and cutout indices.

    Parameters:
    - fil (list): List of filter channels.
    - tile_idx (int): Index of the tile.
    - cutout_idx (int): Index of the cutout.
    - info (bool): If True, return WCS information along with the image data.

    Returns:
    - ndarray: Array of image data.
    - WCS: World Coordinate System information (if info=True).
    """
    img = []
    for f in fil:
        img_path = get_path(tile_idx, f=f)[cutout_idx]
        hdu = load_fits(img_path)
        img.append(hdu.data)
    wcs = WCS(hdu.header)
    if info:
        return np.array(img), wcs
    else:
        return np.array(img)
    
def get_idx_from_tilename(tilename):
    """
    Get the index of a tile from its name.

    Parameters:
    - tilename (str): Name of the tile.

    Returns:
    - int: Index of the tile.
    """
    tilename = tilename.strip()
    cats_only_one = cutclass.cats[cutclass.cats['Filter'] == 'r']
    list(map(str.strip, cats_only_one['Tile name'].data)).index(tilename)

    return list(map(str.strip, cats_only_one['Tile name'].data)).index(tilename)


def get_path(tile_idx, f='r'):
    """
    Get the path of all the fits files in one tile per given filter.

    Parameters:
    - tile_idx (int): Index of the tile.
    - f (str): Filter.

    Returns:
    - list: List of paths to FITS files.
    """
    # Cutclass = Cutouts()
    tile_list = sorted(set(cutclass.cats['Tile name']))  # set!! so it shows only for one filter
    tile_name = tile_list[tile_idx]
    tile_fldr = change_specialcharacters(tile_name)  # tile_name.replace('KIDS', 'KiDS_DR4.0').replace('.', 'p').replace('-', 'm').rstrip(' ') #get nime of the tile as saved
    tile_path = sorted(glob.glob(os.path.join(settings.path_to_save_imgs, f + '_band/*')))  # get the path of all the tiles [!!!]
    tile_idx_saved = index_containing_substring(tile_path, tile_fldr)
    img_path = sorted(glob.glob(tile_path[tile_idx_saved] + '/*'))
    return img_path

def get_rowtab_by_ID(tile_name, tile_ID):
    """
    Get a row from a table based on tile name and tile ID.

    Parameters:
    - tile_name (str): Name of the tile.
    - tile_ID: ID of the tile.

    Returns:
    - DataFrame or Table: Row corresponding to the specified tile name and ID.
    """
    # cutclass = gen_cutouts.Cutouts()
    return cutclass.getTableTile(tile_name)[cutclass.getTableTile(tile_name)['ID'] == tile_ID]

def get_img(fil=settings.channels, tile_idx=None, cutout_idx=None, info=False):
    """
    Load image data from specified tile and cutout indices.

    Parameters:
    - fil (list): List of filter channels.
    - tile_idx (int): Index of the tile.
    - cutout_idx (int): Index of the cutout.
    - info (bool): If True, return WCS information along with the image data.

    Returns:
    - ndarray: Array of image data.
    - WCS: World Coordinate System information (if info=True).
    """
    img = []
    for f in fil:
        img_path = get_path(tile_idx, f=f)[cutout_idx]
        hdu = load_fits(img_path)
        img.append(hdu.data)
    wcs = WCS(hdu.header)
    if info:
        return np.array(img), wcs
    else:
        return np.array(img)

def plot_oneimg_n_channels(img, channels=settings.channels, cmap=None, infotitle=None, name2save=None):
    """
    Plot a single image with multiple channels.

    Parameters:
    - img (ndarray): Image data.
    - channels (list): List of channel names.
    - cmap: Colormap for plotting.
    - infotitle: Title for the plot.
    - name2save: Name to save the plot as a file.
    """
    n_channels = len(channels)
    if np.shape(img)[-1] != n_channels:
        # i suppose that the channels are the first column
        img = np.transpose(img, axes=(1, 2, 0))

    fig, axs = plt.subplots(1, n_channels, figsize=(25, 25), facecolor='w', edgecolor='w')
    fig.subplots_adjust(hspace=.5, wspace=.5)
    axs = axs.ravel()
    if infotitle != None:
        plt.title(infotitle, fontsize=16, x=-2, y=1.1)
    for i in range(n_channels):
        axs[i].imshow(img[:, :, i], cmap=cmap)  # , norm=LogNorm(vmin=np.min(img[:,:,i]), vmax=np.max(img[:,:,i])))
    if name2save != None:
        plt.savefig(name2save, bbox_inches='tight')
    plt.axis('off')
    plt.show(block=False)

def getID(ID, n):
    """
    Get the ID from the name of the file.

    Parameters:
    - ID (list): List of file names.
    - n (int): Index of the file.

    Returns:
    - str: Extracted ID from the file name.
    """
    # get the ID from the name of the file
    return ID[n][0].split('/')[-1][:-5]

def folderexist(path):
    """
    Check if a folder exists and create it if it doesn't.

    Parameters:
    - path (str): Path to the folder.
    """
    if not (os.path.exists(path) or os.path.exists(path + '/')):
        os.mkdir(path)


def createdict_within(mydict, add_dict):
    """
    Create a dictionary with additional key-value pairs.

    Parameters:
    - mydict (dict): Original dictionary.
    - add_dict (dict): Dictionary with additional key-value pairs.

    Returns:
    - dict: Merged dictionary.
    """
    mydict.update(add_dict)
    return mydict


def createdict_withinfo(kidstile, ID_obj, ra, dec, z, mag, mag_err,
                        ext_g, ext_r, g_min_r, r_min_i, z_min, z_max, z_ml, sg2dphot, flags, imaflags):
    """
    Create a dictionary with information.

    Parameters:
    - kidstile (str): KIDS tile.
    - ID_obj (str): Object ID.
    - ra, dec, z, mag, mag_err, ext_g, ext_r, g_min_r, r_min_i, z_min, z_max, z_ml, sg2dphot, flags, imaflags: Object properties.

    Returns:
    - dict: Dictionary with object information.
    """
    mydict = {
        'KIDS_TILE': kidstile,
        'KIDS_ID': ID_obj,
        'RA': ra,
        'DEC': dec,
        'z': z,
        'mag': mag,
        'mag_err': mag_err,
        'ext_g': ext_g,
        'ext_r': ext_r,
        'g_min_r': g_min_r,
        'r_min_i': r_min_i,
        'z_min': z_min,
        'z_max': z_max,
        'z_ml': z_ml,
        'sg2dphot': sg2dphot,
        'flags': flags,
        'imaflags': imaflags
    }
    return mydict

def getinfo_from_catalog_tile(ID_obj, ID_tile):
    """
    Get information from a catalog for a specific object and tile.

    Parameters:
    - ID_obj (str): Object ID.
    - ID_tile (str): Tile ID.

    Returns:
    - dict: Dictionary with object information.
    """
    tile_list = sorted(set(cutclass.cats['Tile name']))

    try:
        catalog_tile = cutclass.getTableTile(ID_tile)
    except Exception as e:
        print(f"Error: {e}")
        print(f"ID_obj: {ID_obj}, ID_tile: {ID_tile}")
        sys.exit()
    idx_catalog = np.where(catalog_tile['ID'] == ID_obj)[0]
    if len(idx_catalog) == 0:
        idx_catalog = np.where(catalog_tile['ID'] == ('KiDSDR4 ' + ID_obj))[0]
    if len(idx_catalog) > 1:
        raise ValueError('Duplicates!')

    z = catalog_tile['Z_B'][idx_catalog].data[0]
    ra = catalog_tile['RAJ2000'][idx_catalog].data[0]
    dec = catalog_tile['DECJ2000'][idx_catalog].data[0]
    kidstile = catalog_tile['KIDS_TILE'][idx_catalog].data[0]
    mag = catalog_tile['MAG_AUTO'][idx_catalog].data[0]
    mag_err = catalog_tile['MAGERR_AUTO'][idx_catalog].data[0]
    ext_g = catalog_tile['EXTINCTION_g'][idx_catalog].data[0]
    ext_r = catalog_tile['EXTINCTION_r'][idx_catalog].data[0]
    g_min_r = catalog_tile['COLOUR_GAAP_g_r'][idx_catalog].data[0]
    r_min_i = catalog_tile['COLOUR_GAAP_r_i'][idx_catalog].data[0]
    z_min = catalog_tile['Z_B_MIN'][idx_catalog].data[0]
    z_max = catalog_tile['Z_B_MAX'][idx_catalog].data[0]
    z_ml = catalog_tile['Z_ML'][idx_catalog].data[0]
    sg2dphot = catalog_tile['SG2DPHOT'][idx_catalog].data[0]
    flags = catalog_tile['Flag'][idx_catalog].data[0]
    imaflags = catalog_tile['IMAFLAGS_ISO'][idx_catalog].data[0]

    mydict = createdict_withinfo(kidstile, ID_obj, ra, dec, z, mag, mag_err,
                                 ext_g, ext_r, g_min_r, r_min_i, z_min, z_max, z_ml, sg2dphot, flags, imaflags)
    return mydict


def set_threshold(df,n_model,th=0.95):
    #print('threshold '+str(th))
    suresure=df[df['prob_'+str(n_model)]>th]
    return suresure


def from_fits_to_array(folder_path, img_ID, tile_ID, channels=settings.channels):
    """
    Converts FITS files to a NumPy array.

    Parameters:
        folder_path (str): Path to the folder containing FITS files.
        img_ID (str): ID of the image.
        tile_ID (str): ID of the tile.
        channels (list): List of channels to consider.

    Returns:
        np.ndarray: A NumPy array containing image data from specified channels.
    """

    # List to store image data from different channels
    img = []

    # Ensure tile_ID is formatted correctly
    tile_ID = change_specialcharacters(tile_ID)

    # If only one channel is specified, convert it to a list
    if len(channels) == 1:
        channels = [channels]

    # Loop through specified channels
    for f in channels:
        try:
            # Attempt to load the FITS file from the first path format
            hdu = load_fits(os.path.join(folder_path, f+'_band', 'tile_' + tile_ID, img_ID + '.fits'))
        except FileNotFoundError:
            try:
                # If the first attempt fails, try the second path format
                hdu = load_fits(os.path.join(folder_path, f+'_band', tile_ID, img_ID + '.fits'))
            except FileNotFoundError:
                # If both attempts fail, handle the exception or add additional attempts as needed
                # Example: hdu = load_fits(os.path.join('/path/to/alternative/location', f+'_band', tile_ID, img_ID + '.fits'))
                print(f'File not found for channel {f} in tile {tile_ID} for image {img_ID}')

        # Append the data to the img list
        img.append(hdu.data)

    # Get the header information from the last successfully loaded FITS file
    wcs = WCS(hdu.header)

    # Convert the list of images to a NumPy array
    return np.array(img)

    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_tile_from_coords(df_survey,ra_col_name='RAdeg',dec_col_name='DEdeg'):
    
    '''
    dec_col_name: name of the column dec in the survey 
    ra_col_name: name of the column ra in the survey 
    '''
    #cuclass = gen_cutouts.Cutouts()
    df_nodup = cutclass.cats[cutclass.cats['Filter'] == 'r'] #KiDs obs table

    idx_tileinkids = []
    idx_insurvey = []


    for i in range(len(df_nodup)):
        idx = get_tiles_ra_dec_range(df_survey, df_nodup['DEC'][i],df_nodup['DEC'][i]+1,df_nodup['RA'][i],df_nodup['RA'][i]+1 , 
                                                     dec_col_name=dec_col_name,ra_col_name=ra_col_name)
        #print(idx)
        if len(idx)>0:
            idx_tileinkids.extend(np.ones(len(idx), dtype=np.int8)*i)
            idx_insurvey.extend(idx)
    if len(idx_tileinkids)!=len(idx_insurvey):
        raise ValueError('Different lenght')
    print(len(idx_tileinkids),len(idx_insurvey)    )
    print('I found {} obj in the KiDs survey'.format(len(idx_tileinkids)))
    return idx_tileinkids,idx_insurvey
    
