# Author: Jasper Ristkok
# v1.0

# Code to batch-convert all echellograms (photos) in a folder to spectrum using
# the calibration data from _Echellogram-to-spectrum-calibration.py


##############################################################################################################
# CONSTANTS
##############################################################################################################

# If working_path is None then the same folder will be used for all inputs and outputs where the code is executed
# If you aren't using Windows then paths should have / instead of \\ (double because \ is special character in strings)
working_path = None

system_path_symbol = '\\' # / for Unix, code checks for it later

##############################################################################################################
# IMPORTS
##############################################################################################################

import numpy as np
import math
import os
import re # regex
import json
from PIL import Image as PILImage
import platform


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
    
    files = get_files_in_folder(working_path)
    for filename in files:
        try:
            process_photo(filename, calibration_data)
        except Exception as e:
            print(e)
            print_crash_tree(e)
            
    

##############################################################################################################
# Preparation phase
##############################################################################################################

# if program is in automatic mode (you don't execute it for a Python IDE and change the code here) then 
# use all input and output paths as the folder where the script (.py file) is executed from
def get_path():
    
    # root path as the location where the python script was executed from 
    # (put the script .py file into the folder with the Echellograms)
    global system_path_symbol, working_path
    
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
def get_files_in_folder(path):
    
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


##############################################################################################################
# Processing phase
##############################################################################################################

def process_photo(filename, calibration_data):
    filepath = working_path + filename
    photo_array = load_photo(filepath)
    
    # negative values to 0
    photo_array[photo_array < 0] = 0
    
    # Get spectrum x and y values
    x_values, y_values = compile_spectrum(photo_array, calibration_data)
    
    # save the spectrum into txt file with ; delimiter (same as Sophi nXt)
    save_spectrum(filename, x_values, y_values)



# Get z-values of corresponding x and y values and cut them according to left and right bounds
def compile_spectrum(photo_array, calibration_data):
    x_values = []
    z_values = []
    
    
    nr_pixels = photo_array.shape[0]
    photo_pixels = np.arange(nr_pixels)
    
    # Compile z-data from the photo with all diffraction orders
    # Iterate over orders backwards because they are sorted by order nr (top to bottom)
    for order_idx in range(len(calibration_data['dynamic']) - 1, -1, -1):
        
        order = calibration_data['dynamic'][order_idx]
        curve_array, poly_coefs = get_polynomial_points(order, nr_pixels)
        
        curve_x = photo_pixels
        curve_y = curve_array
        curve_y = np.round(curve_y)
        y_pixels = curve_y.astype(np.int32) # convert to same format as x-values
        
        
        # Get bounds for this diffraction order
        [x_left, x_right] = calibration_data['static'][order_idx]['bounds_px']
        
        # Default bounds as first px and last px
        if x_left is None:
            x_left = 0
        if x_right is None:
            x_right = nr_pixels - 1
        
        # get z-values corresponding to the interpolated pixels on the order curve
        for idx2 in photo_pixels:
            
            # Save px only if it's between left and right bounds
            x_px = curve_x[idx2]
            
            # Do stuff if point is between bounds
            if x_left <= x_px <= x_right:
                y_px = y_pixels[idx2]
                
                [wave_start, wave_end] = calibration_data['static'][order_idx]['bounds_wave']
                
                # No input with wavelengths, output x-axis as pixel values
                if (wave_start is None) or (wave_end is None):
                    raise Exception('Calibration data doesn\'t have wavelength with order: ' + str(order_idx))
                    
                
                # Has input with wavelengths, output x-axis as wavelengths
                else:
                    x_value = linear_regression(x_px, x_left, x_right, wave_start, wave_end)
                    x_values.append(x_value)
                    
                
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
                
    return x_values, z_values

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


# save the spectrum into txt file with ; delimiter (same as Sophi nXt)
def save_spectrum(filename, x_values, y_values):
    
    # get filename without .tif
    pattern = r'(.+?)\.tif$'
    regex_result = re.search(pattern, filename)
    identificator = regex_result[1]
    
    filepath = working_path + identificator + '.txt'
    
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