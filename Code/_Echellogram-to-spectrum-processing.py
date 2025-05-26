# Author: Jasper Ristkok
# v1.1

'''
Code to batch-convert all echellograms (photos) in a folder to spectrum using
the calibration data from _Echellogram-to-spectrum-calibration.py
The code assumes that the calibration file is formatted correctly.

Inputs:
    * _Calibration_data.json
    * _Correction_multipliers.csv (optional)

Outputs:
    * spectra in _Raw_spectra folder OR in _Corrected_spectra folder
    Folder name depends on whether _Correction_multipliers.csv exists

'''

##############################################################################################################
# CONSTANTS
##############################################################################################################

# If working_path is None then the same folder will be used for all inputs and outputs where the code is executed
# If you aren't using Windows then paths should have / instead of \\ (double because \ is special character in strings)
working_path = None
system_path_symbol = '\\' # Gets checked later

##############################################################################################################
# IMPORTS
##############################################################################################################

import math
import os
import re # regex
import json
import platform

import numpy as np
from PIL import Image as PILImage
from scipy.interpolate import interp1d

##############################################################################################################
# DEFINITIONS
# Definitions are mostly structured in the order in which they are called at first.
##############################################################################################################


def main_program():
    
    # Code is executed from folder with Echellograms
    if working_path is None:
        get_path()
    
    calibration_data = load_calibration_data()
    if calibration_data is None:
        raise Exception('No calibration data!')
        
    # Get the output spectrum wavelengths (used here for interpolation)
    wavelengths = get_spectrum_wavelengths(calibration_data)
    
    # If folder contains spectral sensitivity multipliers then it's written into this variable
    multipliers = get_correction_multipliers(wavelengths)
    
    # Create folder for output, name depends on whether correction file exists
    create_folder(multipliers)
    
    files = get_photos_in_folder(working_path)
    count = 0
    for filename in files:
        try:
            process_photo(filename, calibration_data, wavelengths, multipliers)
        except Exception as e:
            print(e)
            print_crash_tree(e)
        
        # Feedback on progress
        count += 1
        if (count % 20) == 0:
            print(filename + ' done')
    
    
    print('Program finished')
    

##############################################################################################################
# Preparation phase
##############################################################################################################

# if program is in automatic mode (you don't execute it for a Python IDE and change the code here) then 
# use all input and output paths as the folder where the script (.py file) is executed from
def get_path():
    
    # root path as the location where the python script was executed from 
    # (put the script .py file into the folder with the Echellograms)
    global working_path, system_path_symbol
    
    # Get OS name for determining path symbol
    if platform.system() == 'Windows':
        system_path_symbol = '\\'
    else:
        system_path_symbol = '/' # Mac, Linux
    
    # Get folder where .py file was executed from
    working_path = os.getcwd() # 'd:\\github\\echellogram-to-spectrum\\code'
    working_path += system_path_symbol

def load_calibration_data():
    calibration_data = None
    
    # calibration done previously, load that data
    with open(working_path + '_Calibration_data.json', 'r') as file: # allow crashing if no file
        calibration_data = json.load(file)
    
    first_order_nr = int(calibration_data['first_order_nr'])
    
    # Remove irrelevant orders
    for idx in range(len(calibration_data['dynamic'])):
        
        order_nr = int(calibration_data['static'][idx]['order_nr'])
        if (order_nr < first_order_nr) or (not calibration_data['static'][idx]['use_order']):
            del calibration_data['static'][idx]
    
    return calibration_data

# Get all .tif files in folder
def get_photos_in_folder(path):
    
    filenames = [f for f in os.listdir(working_path) if os.path.isfile(os.path.join(working_path, f))]
    if len(filenames) == 0:
        raise Exception('Error: no files in input folder: ' + str(working_path))
    
    pattern = '\.tif$'
    filenames = [f for f in filenames if re.search(pattern, f)]
    
    return filenames

def load_photo(filepath):
    image = PILImage.open(filepath)
    imArray = np.array(image)
    return imArray


# Return an array of wavelengths for the spectrum
def get_spectrum_wavelengths(calibration_data):
    wavelengths = []
    
    # Compile pixels and thus wavelengths from the photo with all diffraction orders
    # Iterate over orders backwards because they are sorted by order nr (top to bottom)
    for order_idx in range(len(calibration_data['dynamic']) - 1, -1, -1):
        
        # Get bounds for this diffraction order
        [x_left, x_right] = calibration_data['static'][order_idx]['bounds_px']
        [wave_start, wave_end] = calibration_data['static'][order_idx]['bounds_wave']
        
        # No input with wavelengths, output x-axis as pixel values
        if (wave_start is None) or (wave_end is None):
            order_nr = calibration_data['static'][order_idx]['order_nr']
            raise Exception('Calibration data doesn\'t have wavelength with order: ' + str(order_nr))
        
        # integers for indices
        x_left = int(x_left)
        x_right = int(x_right)
        
        # output x-axis as wavelengths between the order bounds
        for x_px in range(x_left, x_right + 1):
            wavelength = linear_regression(x_px, x_left, x_right, wave_start, wave_end)
            wavelengths.append(wavelength)
    
    wavelengths = np.array(wavelengths)
    return wavelengths


# Load correction multipliers from file if it exists, otherwise return None
def get_correction_multipliers(wavelengths):
    try:
        multipliers_array = np.loadtxt(working_path + '_Correction_multipliers.csv', delimiter = ',', skiprows = 1)
    except:
        print('_Correction_multipliers.csv not found')
        return None
    
    # Interpolate the multipliers to match the wavelengths of output spectra
    wavelengths_original = multipliers_array[:, 0]
    multipliers_original = multipliers_array[:, 1]
    
    # Create linear interpolator with extrapolation enabled
    linear_interp = interp1d(wavelengths_original, multipliers_original,
        kind='quadratic', fill_value='extrapolate', assume_sorted=False)
    
    # Interpolate multipliers at the spectrum wavelengths
    interpolated_multipliers = linear_interp(wavelengths)
    return interpolated_multipliers


# Create output folder if it doesn't exist
def create_folder(multipliers):
    
    # Get subfolder name depending on whether correction multipliers file exists
    folder_name = 'Raw_spectra' + system_path_symbol
    if not multipliers is None:
        folder_name = 'Corrected_spectra' + system_path_symbol
    
    path = working_path + folder_name
    if not os.path.isdir(path):
        os.mkdir(path)



##############################################################################################################
# Processing phase
##############################################################################################################

def process_photo(filename, calibration_data, wavelengths, multipliers):
    filepath = working_path + filename
    photo_array = load_photo(filepath)
    
    # negative values to 0
    #photo_array[photo_array < 0] = 0 # keep negatives to avoid noise bias
    
    # Get spectrum x and y values
    y_values = compile_spectrum(photo_array, calibration_data)
    
    # Use spectral sensitivity multipliers
    y_values = apply_correction(y_values, multipliers)
    
    
    # save the spectrum into txt file with ; delimiter (same as Sophi nXt)
    save_spectrum(filename, wavelengths, y_values, multipliers)

# Get z-values of corresponding x and y values and cut them according to left and right bounds
def compile_spectrum(photo_array, calibration_data):
    z_values = []
    
    nr_pixels = photo_array.shape[0]
    
    # Compile z-data from the photo with all diffraction orders
    # Iterate over orders backwards because they are sorted by order nr (top to bottom)
    for order_idx in range(len(calibration_data['dynamic']) - 1, -1, -1):
        
        order = calibration_data['dynamic'][order_idx]
        curve_y, poly_coefs = get_polynomial_points(order, nr_pixels)
        
        curve_y = np.round(curve_y) # Convert to y-coordinates/indices
        y_pixels = curve_y.astype(np.int32) # convert to same format as x-values
        
        
        # Get bounds for this diffraction order
        [x_left, x_right] = calibration_data['static'][order_idx]['bounds_px']
        [wave_start, wave_end] = calibration_data['static'][order_idx]['bounds_wave']
        
        # No input with wavelengths, output x-axis as pixel values
        if (wave_start is None) or (wave_end is None):
            order_nr = calibration_data['static'][order_idx]['order_nr']
            raise Exception('Calibration data doesn\'t have wavelength with order: ' + str(order_nr))
        
        # Default bounds as first px and last px
        if x_left is None:
            x_left = 0
        if x_right is None:
            x_right = nr_pixels - 1
        
        # integers for indices
        x_left = int(x_left)
        x_right = int(x_right)
        
        # Save px only if it's between left and right bounds
        # get z-values corresponding to the interpolated pixels on the order curve
        for idx2 in range(x_left, x_right + 1):
            x_px = idx2 # photo pixels have same values as indices
            y_px = y_pixels[idx2]
            
            # Get the width to integrate over (between two diffraction orders)
            width = 1 
            center = get_avg_y(calibration_data['dynamic'][order_idx])
            if order_idx > 0: # check for out of bounds error
                low = get_avg_y(calibration_data['dynamic'][order_idx - 1])
                width = (center - low) / 2
            elif len(calibration_data['dynamic']) > order_idx + 1: # check for out of bounds error
                high = get_avg_y(calibration_data['dynamic'][order_idx + 1])
                width = (high - center) / 2
            
            # very important, otherwise 2.49 and 2.51 will have 150% jump in integral because of rounding
            width = clip(width, min_v = 3) 
            
            
            # Get z value from Echellogram
            integral = integrate_order_width(photo_array, x_px, y_px, width = width)
            z_values.append(integral) # x and y have to be switched (somewhy)
    
    return z_values


# Gets average y-value of points in order
def get_avg_y(points):
    array = []
    for point in points:
        array.append(point['y'])
    
    return np.average(array)


def gather_points(order_points):
    xlist = []
    ylist = []
    for idx in range(len(order_points)):
        xlist.append(order_points[idx]['x'])
        ylist.append(order_points[idx]['y'])
    return xlist, ylist


# Sum pixels around the order
# If the width is even number then take asymmetrically one pixel from lower index
# if use_weights is True then the integral is summed with Gaussian weights (max is in center) with FWHM of width
def integrate_order_width(photo_array, x_pixel, y_pixel, width = 1, use_weights = False):
    width = round(width)
    
    if width == 1:
        return photo_array[y_pixel, x_pixel] # x and y have to be switched (somewhy)
    
    integral = 0
    x_pixel = clip(x_pixel, 0, photo_array.shape[0] - 1)
    idx_lower = math.floor(width / 2)
    idx_higher = math.ceil(width / 2) # range omits last value
    for y_idx in range(y_pixel - idx_lower, y_pixel + idx_higher):
        y_idx = clip(y_idx, 0, photo_array.shape[0] - 1)
        
        gaussian_weight = 1
        if use_weights:
            a = 1 # normalized
            c = width / 2.35482 # width is FWHM
            gaussian_weight = a * math.exp(-(y_pixel - y_idx) ** 2 / 2 / c ** 2)
        
        integral += photo_array[y_idx, x_pixel] * gaussian_weight # x and y have to be switched (somewhy)
    
    return integral


# Apply spectral sensitivity multipliers
def apply_correction(y_values, multipliers):
    if multipliers is None:
        return y_values
    
    # Multiply arrays row by row
    y_values = np.array(y_values) # convert from list to array
    corrected_values = y_values * multipliers
    return corrected_values


# save the spectrum into txt file with ; delimiter (same as Sophi nXt)
def save_spectrum(filename, x_values, y_values, multipliers):
    
    # get filename without .tif
    pattern = r'(.+?)\.tif$'
    regex_result = re.search(pattern, filename)
    identificator = regex_result[1]
    
    # Get subfolder name depending on whether correction multipliers file exists
    folder_name = 'Raw_spectra' + system_path_symbol
    if not multipliers is None:
        folder_name = 'Corrected_spectra' + system_path_symbol
    
    filepath = working_path + folder_name + identificator + '.txt'
    
    array = np.column_stack((x_values, y_values))
    np.savetxt(filepath, array, delimiter = ';', fmt = "%.8f")

##############################################################################################################
# Utility functions
##############################################################################################################

# Do linear regression and find the y-value at provided x
def linear_regression(x, x_start, x_end, y_start, y_end):
    
    dx = x_end - x_start
    dy = y_end - y_start
    slope = dy / dx
    intercept = y_start - slope * x_start
    
    y = x * slope + intercept
    return y
    


# Calculate points of polynomial for plotting
def get_polynomial_points(order, arr_length):
    xlist, ylist = gather_points(order)
    
    # get regression coeficients
    poly_coefs = np.polynomial.polynomial.polyfit(xlist, ylist, 2)
    
    # get array of points on image for the polynomial
    curve_array = np.empty(arr_length)
    for idx in range(arr_length):
        curve_array[idx] = poly_func_value(idx, poly_coefs)
    
    # clip the values
    curve_array = np.clip(curve_array, 0, arr_length - 1)
    
    return curve_array, poly_coefs


# Return a value for the polynomial
def poly_func_value(x_value, coefs):
    value = 0
    for idx, coef in enumerate(coefs):
        value += coef * x_value ** idx
    return round(value, 10)

def clip(value, min_v = -math.inf, max_v = math.inf):
    return max(min_v, min(value, max_v))


def print_crash_tree(excep):
    
    # Print where exactly the exeption was raised
    print("Exception occurred in functiontree:")
    tb = excep.__traceback__
    while tb.tb_next:  # Walk to the last traceback frame (where exception was raised)
        print(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    print(tb.tb_frame.f_code.co_name)



##############################################################################################################
# RUN MAIN PROGRAM
##############################################################################################################
main_program()