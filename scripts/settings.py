import configparser
import os
# Read the config.ini file
config = configparser.ConfigParser()
# Get the path of the current script
script_path = os.path.abspath(__file__)
config.read(os.path.join(os.path.dirname(script_path),'config.ini'))

# Access the global variables
tiles_csv = config.get('GlobalVariables', 'tiles_csv')
path_kids_data = config.get('GlobalVariables', 'path_kids_data')
path_to_save_imgs = config.get('GlobalVariables', 'path_to_save_imgs')
path_weights = config.get('GlobalVariables', 'path_weights')
path_lens_foundby_kids = config.get('GlobalVariables', 'path_lens_foundby_kids')
catalog_lens_kidscollab = config.get('GlobalVariables', 'catalog_lens_kidscollab')
channels = config.get('GlobalVariables', 'channels').split(', ')
threshold = config.getfloat('GlobalVariables', 'threshold')
z_min = config.getfloat('GlobalVariables', 'z_min')
z_max = config.getfloat('GlobalVariables', 'z_max')
tile_idx = list(map(int, config.get('GlobalVariables', 'tile_idx').split(', ')))
tile_not_seen = list(map(int, config.get('GlobalVariables', 'tile_not_seen').split(', ')))
generate_cutouts_and_find_lenses = config.getboolean('GlobalVariables', 'generate_cutouts_and_find_lenses')
folders_retrain = list(map(str.strip, config.get('GlobalVariables', 'folders_retrain').split(',')))
folders_label = list(map(str.strip, config.get('GlobalVariables', 'folders_label').split(',')))