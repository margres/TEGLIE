import os
import numpy as np
from . import settings
from .utils import change_specialcharacters
from  .pyplz_rgbtools  import marshall16_pil_format, lupton04_pil_format
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.io import fits


# Default RGB bands and settings
rgbbands = ('i', 'r', 'g') 
scales = np.array([0.5, 0.6, 1.8]) / 10.**(-2./5.*27.)
alpha = 1.
Q = 1.

def display_png(path_png):
    """
    Display PNG image from the given path.

    Parameters:
        path_png (str): Path to the PNG image file.
    """
    print(path_png)
    img = mpimg.imread(path_png)
    imgplot = plt.imshow(np.flip(img, axis=0))
    plt.show()

def has_dimension_three(arr):
    """
    Check if the input array has three dimensions.

    Parameters:
        arr (numpy.ndarray): Input array.

    Returns:
        bool: True if the array has three dimensions, False otherwise.
    """
    return arr.ndim >= 3 and 3 in arr.shape

def make_rgb_one_image(img_array=None , img_name=None, tile_name=None, 
                       folder_cutouts=settings.path_to_save_imgs, display_plot=False, 
                       return_img=False, type_plot='lupton', save_img=False, path_to_save=None ):
    
    """
    Create an RGB image from FITS files or provided image array.

    Parameters:
        img_array (numpy.ndarray): Image data array. If not provided, img_name and tile_name must be provided.
        img_name (str): ID of the image.
        tile_name (str): ID of the tile.
        folder_cutouts (str): Path to the folder containing FITS files.
        display_plot (bool): Whether to display the RGB image plot.
        return_img (bool): Whether to return the RGB image as a NumPy array.
        type_plot (str): Type of RGB plot ('marshall' or 'lupton').
        save_img (bool): Whether to save the RGB image.
        path_to_save (str): Path to save the RGB image if save_img is True.

    Returns:
        np.ndarray: RGB image as a NumPy array if return_img is True.
    """

    if img_array is None:
        if (img_name is not None) and (tile_name is not None):
            # Load image from file
            if not img_name.endswith('.fits'):
                img_name = img_name+'.fits'

            rgbdata = []
            for band in rgbbands:
                try:
                    img_to_display = os.path.join(folder_cutouts, band+'_band', 'tile_'+change_specialcharacters(tile_name), img_name)
                    img_array = fits.getdata(img_to_display, memmap=True)
                except:
                    try:
                        img_to_display = os.path.join(folder_cutouts, band+'_band', change_specialcharacters(tile_name), img_name)
                
                    except FileNotFoundError:
                        print('in HD_MG')
                        img_to_display = os.path.join('/home/grespanm/mnt/HD_MG/KiDS_cutout', band+'_band', change_specialcharacters(tile_name), img_name)

                img_array = fits.getdata(img_to_display, memmap=True)
                rgbdata.append(img_array)
             
        else:
            raise ValueError("Either 'img_name' and 'tile_name' or 'img_array' must be provided.")
    else:
    
        # Image array is already provided
        if not isinstance(img_array, np.ndarray):
            raise ValueError("The 'img_array' argument must be a NumPy array.")
        if has_dimension_three(img_array):
            pass
        else:
            #raise ValueError('img needs 3 dimensions and channels in the order (i, r, g)')
            dim_idx = np.where(np.array(img_array.shape) == 4)[0]
            if dim_idx==0:
                img_array = img_array[:3,:,:]
            elif dim_idx==2:
                img_array = img_array[:,:,:3]

        # Find the index of the first dimension that has length 3
        dim_idx = np.where(np.array(img_array.shape) == 3)[0]

        #i want shape(3,101,101)
        if dim_idx==2:
            img_array = np.transpose(img_array, (2,0,1))
        elif dim_idx==0:
           
            pass

        # Create a new array with the third dimension in the order i,r,g
        rgbdata = img_array[[1, 0, 2], :, :]
        img_array = img_array[0,:,:]

    s = img_array.shape
   
    im = Image.new('RGB', (s[1], s[0]), 'black')
    if type_plot== 'marshall':
        im.putdata(marshall16_pil_format(rgbdata, scales=scales, alpha=alpha, Q=Q))
    elif type_plot=='lupton':
        im.putdata(lupton04_pil_format(rgbdata, scales=scales))
    else:
        raise ValueError('type_plot error, available values marshall and lupton')
    #if (img_name is not None) and (tile_name is not None):
    if save_img:
        if (img_name is None) and (tile_name is None):
            raise ValueError('If you want to save the image give tile name and image name!')
        if path_to_save is None:
            raise ValueError('If you want to save the image you need to give the path (parent directory) to save it.')
        name_to_save =tile_name+'_ID_'+'.'.join(img_name.split('.')[:-1])
        pp_save=os.path.join(path_to_save,'RGB_images')

        if not os.path.exists(pp_save):
            os.mkdir(pp_save)
        pp_save = os.path.join(pp_save, name_to_save+'.png')
        im.save(pp_save)

    if display_plot:
        fig, ax = plt.subplots(figsize=(5, 5)) 
        # Convert the PIL image to a NumPy array
        np_image = np.array(im)
        # Display the NumPy array as an image with Matplotlib
        plt.imshow(np_image)
        plt.axis('off')
        plt.show()

    if return_img:
        return np.array(im)