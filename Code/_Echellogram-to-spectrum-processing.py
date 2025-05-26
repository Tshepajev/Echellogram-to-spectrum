# Author: Jasper Ristkok
# v2.0

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

# TODO: resample spectrum because multiple same/similar wavelengths cause issues with interpolation
# TODO: implement user multipliers properly


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
import copy

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
    
    # Class for sharing variables between functions and easier copy-paste from Calibrator code
    processor_class()
    
    print('Program finished')
    

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


##############################################
# Classes
##############################################

# Generic point class, a point on a graph
class point_class():
    def __init__(self, x, y = None, z = None, group = None):
        
        if x is None:
            raise ValueError('No x data')
        
        self.x = x
        self.y = y
        self.z = z
        self.group = group


class static_calibration_data():
    def __init__(self, order_nr = -1, existing_data = None):
        self.order_nr = order_nr
        self.bounds_px = [None, None] # pixel index after shift
        self.bounds_wave = [None, None] # wavelengths after shift
        self.bounds_middle = [None, None] # 3rd pixel and 3rd wavelength from the middle of the order after shift
        self.bounds_px_original = [None, None] # pixel index from input file
        self.bounds_wave_original = [None, None] # wavelengths from input file
        self.bounds_middle_orig = [None, None] # 3rd pixel and 3rd wavelength from the middle of the order
        self.use_order = True
        
        self.load_encode(existing_data)
    
    # Convert saved dictionary into point classes
    def load_encode(self, existing_data):
        self.order_nr = existing_data['order_nr']
        self.bounds_px = existing_data['bounds_px']
        self.bounds_wave = existing_data['bounds_wave']
        self.bounds_px_original = existing_data['bounds_px_original']
        self.bounds_wave_original = existing_data['bounds_wave_original']
        self.use_order = existing_data['use_order']
        
        # backwards compatibility
        keys = existing_data.keys()
        if 'bounds_middle' in keys:
            self.bounds_middle = existing_data['bounds_middle']
            self.bounds_middle_orig = existing_data['bounds_middle_orig']

# Line of a diffraction order (horizontal), x is horizontal and y vertical pixels
class order_points():
    def __init__(self, image_width = 1, existing_data = None):
        self.image_width = image_width
        self.avg_y = 0
        self.xlist = None
        self.ylist = None
        self.points = []
        
        self.load_encode(existing_data)
        self.update()
    
    
    def update(self):
        self.create_lists()
        self.avg_y = np.average(self.ylist)
    
    def create_lists(self):
        self.xlist = self.to_list(self.points, 'x')
        self.ylist = self.to_list(self.points, 'y')
    
    # Take axis string as argument and return a list of the axis values of the points
    def to_list(self, points, axis):
        array = []
        for point in self.points:
            array.append(getattr(point, axis, None))
        return array
    
    
    # Convert saved dictionary into point classes
    def load_encode(self, existing_data):
        self.points = []
        
        # Iterate over points and encode them into classes
        for point_dict in existing_data:
            self.points.append(point_class(point_dict['x'], y = point_dict['y']))
        
        self.update()
    

# Class for sharing variables between functions and easier copy-paste from Calibrator code
class processor_class():
    
    def __init__(self):
        self.init_class_variables()
        self.main_code()
    
    def init_class_variables(self):
        self.working_path = working_path
        
        self.static_idx_offset = 0 # difference between first_order_nr and smallest order nr in static bounds data
        self.first_order_nr = None
        
        
        self.photo_array = None
        self.cumulative_sum = None # for faster vertical integral calculation
        self.order_poly_coefs = []
        self.max_x_idx = None
        self.max_y_idx = None
        
        # Spectrum data
        self.orders_x_pixels = []
        self.orders_wavelength_coefficients = []
        self.orders_wavelengths = []
        self.orders_intensities = []
        self.compiled_wavelengths = None
        self.compiled_intensities = None
        self.multipliers = None
        
        # lists to store calibration data like diffraction order class instances
        self.calib_data_static = [] # holds order bounds and shift data
        self.calib_data_dynamic = [] # holds points that make up the drawn curves, the order of point classes can change
    
    def main_code(self):
        
        #################################
        # Preparation phase
        #################################
        
        self.load_calibration_data()
        if (len(self.calib_data_static) == 0) or (len(self.calib_data_dynamic) == 0):
            raise Exception('No calibration data!')
        
        # delete the orders and static data which have use_order == False
        #self.remove_irrelevant_orders()
        
        filenames = self.get_photos_in_folder(working_path)
        
        # Get the array size of echellograms
        self.check_tif_array(filenames)
        
        # Get the output spectrum wavelengths (used here for interpolation)
        self.update_spectrum_data_wavelengths()
        
        # Create one array of wavelengths from all orders
        self.compile_wavelengths()
        
        self.update_order_poly_coefs()
        
        # If folder contains spectral sensitivity multipliers then it's written into this variable
        self.get_correction_multipliers()
        
        # Create folder for output, name depends on whether correction file exists
        self.create_folder()
        
        
        #################################
        # Processing phase
        #################################
        
        count = 0
        for filename in filenames:
            try:
                self.process_photo(filename)
            except Exception as e:
                print(e)
                print_crash_tree(e)
            
            # Feedback on progress
            count += 1
            if (count % 20) == 0:
                print(filename + ' done')
    
    ##############################################################################################################
    # Preparation phase
    ##############################################################################################################
    
    # Load data from Calibrator's .json file and find static_idx_offset
    # Shift info is irrelevent, since the dynamic and static lists already contain shifted values
    def load_calibration_data(self):
        
        # calibration done previously, load that data
        json_filenames = get_folder_files_pattern(working_path, pattern = r'^_Calibration_data.*\.json$')
        
        if len(json_filenames) == 0:
            raise Exception('_Calibration_data*.json not found')
        
        if len(json_filenames) > 1:
            print('Warning! More than one .json calibration file detected.')
        
        json_filename = json_filenames[0]
        with open(working_path + json_filename, 'r') as file: # allow crashing if no file
            calibration_data = json.load(file)
        
        
        self.first_order_nr = calibration_data['first_order_nr']
        
        # Encode dictionary into data classes
        self.load_dynamic_data(calibration_data)
        self.load_static_data(calibration_data)
        
        self.update_static_idx_offset()
    
    def load_dynamic_data(self, calibration_data):
        for order_raw in calibration_data['dynamic']:
            self.calib_data_dynamic.append(order_points(existing_data = order_raw))
    
    def load_static_data(self, calibration_data):
        for order_raw in calibration_data['static']:
            order_static_class = static_calibration_data(existing_data = order_raw)
            self.calib_data_static.append(order_static_class)
    
        
    # Offset of index between dynamic and static data (drawn order amount not same as in bounds data)
    def update_static_idx_offset(self):
        min_static_order_nr = math.inf
        
        for data in self.calib_data_static:
            if data.order_nr < min_static_order_nr:
                min_static_order_nr = data.order_nr
        
        self.static_idx_offset = self.first_order_nr - min_static_order_nr
    
    # Get static data element corresponding to the dynamic data index
    def get_static_data(self, dynamic_idx):
        static_idx = dynamic_idx + self.static_idx_offset
        max_static_idx = len(self.calib_data_static) - 1
        
        # Use first or last static data element when out of bounds
        if static_idx < 0:
            new_obj = copy.deepcopy(self.calib_data_static[0])
            new_obj.order_nr += static_idx
            return new_obj
        elif static_idx > max_static_idx:
            new_obj = copy.deepcopy(self.calib_data_static[max_static_idx])
            new_obj.order_nr += (static_idx - max_static_idx) 
            return new_obj
        
        return self.calib_data_static[static_idx]
    
    '''
    # Delete the orders and static data which have use_order == False
    def remove_irrelevant_orders(self):
        
        # Count how many static datapoints have use_order == False in the beginning and update self.first_order_nr
        for static_idx in range(len(self.calib_data_static)):
            static_data = self.calib_data_static[static_idx]
            if static_data.use_order:
                self.first_order_nr = static_data.order_nr # now it means first order that is actually used
                break
        
        # Iterate over dynamic data backwards to avoid index conflicts due to deletion
        for order_idx in range(len(self.calib_data_dynamic) - 1, -1, -1):
            static_idx = order_idx + self.static_idx_offset
            static_data = self.get_static_data(order_idx)
            
            # Delete if use_order == False
            if not static_data.use_order:
                print(f'Skipping order nr {static_data.order_nr}')
                
                del self.calib_data_dynamic[order_idx]
                
                # If the referenced order isn't extrapolated from existing data (exists in data)
                if (static_idx >= 0) and (static_idx <= len(self.calib_data_static) - 1):
                    del self.calib_data_static[static_idx]
    '''
    
    # Get all .tif files in folder
    def get_photos_in_folder(self, path):
        
        filenames = [f for f in os.listdir(working_path) if os.path.isfile(os.path.join(working_path, f))]
        if len(filenames) == 0:
            raise Exception('Error: no files in input folder: ' + str(working_path))
        
        pattern = '\.tif$'
        filenames = [f for f in filenames if re.search(pattern, f)]
        
        return filenames

    
    # Read a random tif file and check the array size of the echellograms
    def check_tif_array(self, filenames):
        filepath = working_path + filenames[0]
        array = self.load_photo(filepath)
        shape = array.shape
        self.max_x_idx = shape[1] - 1
        self.max_y_idx = shape[0] - 1 # y means up-down which means different rows, therefore first index
    
    def load_photo(self, filepath):
        image = PILImage.open(filepath)
        imArray = np.array(image)
        return imArray
    
    
    # Iterate over all orders and update/initialize the data associated with the spectrum (wl and int etc.)
    def update_spectrum_data_wavelengths(self):
        for order_idx in range(len(self.calib_data_dynamic)):
            static_data = self.get_static_data(order_idx)
            
            # Ignore if use_order == False
            if static_data.use_order:
                self.update_order_spectrum_data_wavelengths(order_idx)
            
            # Write empty arrays to keep indexing
            else:    
                print(f'Skipping order nr {static_data.order_nr}')
                self.orders_x_pixels.append(np.empty(0))
                self.orders_wavelength_coefficients.append(np.empty(0))
                self.orders_wavelengths.append(np.empty(0))
            
    
    # Update/initialize the data associated with the spectrum (wl and int etc.) for the given order
    def update_order_spectrum_data_wavelengths(self, order_idx):
        # Get x_pixel values between bounds (e.g. array of ints from 480 to 701)
        self.update_spectrum_data_list(order_idx, self.orders_x_pixels, self.calc_order_x_pixels)
        
        # Calculate coefficients which are used to convert px => wavelength
        self.update_spectrum_data_list(order_idx, self.orders_wavelength_coefficients, self.calc_order_wavelength_coefs)
        
        # Get wavelengths corresponding to the x_pixels
        self.update_spectrum_data_list(order_idx, self.orders_wavelengths, self.calc_order_wavelengths)
        
    
    # Either initialize or update spectrum data lists inline (values_fn output goes into array)
    def update_spectrum_data_list(self, order_idx, array, values_fn):
        list_len = len(array)
        if order_idx > list_len:
            raise Exception(f'update_spectrum_data_list() | Trying to modify uninitialized order_idx {order_idx} when length is {list_len}. function: {values_fn}')
        
        values = values_fn(order_idx)
        if order_idx == list_len: # Initialize
            array.append(values)
        else: # Update
            array[order_idx] = values
    
    # Get x-pixel values for current bounds (e.g. array of ints from 480 to 701)
    def calc_order_x_pixels(self, order_idx):
        if order_idx > len(self.calib_data_dynamic):
            raise Exception(f'calc_order_x_pixels() | order_idx out of bounds: {order_idx}')
        
        # Get bounds for this diffraction order
        [x_left, x_right] = self.get_static_data(order_idx).bounds_px
        
        # Default bounds as first px and last px
        if x_left is None:
            x_left = 0
        if x_right is None:
            x_right = self.max_x_idx
        
        px_values = np.arange(x_left, x_right + 1, dtype = int)
        return px_values
    
    # Calculate coefficients for calculating px => wl
    def calc_order_wavelength_coefs(self, order_idx):
        if order_idx > len(self.calib_data_dynamic):
            raise Exception(f'calc_order_wavelength_coefs() | order_idx out of bounds: {order_idx}')
        
        # Take bounds from (latest aka shifted) order points class
        [px_start, px_end] = self.get_static_data(order_idx).bounds_px
        [wave_start, wave_end] = self.get_static_data(order_idx).bounds_wave
        [px_middle, wave_middle] = self.get_static_data(order_idx).bounds_middle
        
        # Do linear regression and find the wavelength of x_px
        if (px_middle is None) or (wave_middle is None):
            linear_coefs = np.polynomial.polynomial.polyfit([px_start, px_end], [wave_start, wave_end], 1)
            return linear_coefs
        
        # Use quadratic fn
        else:
            poly_coefs = np.polynomial.polynomial.polyfit([px_start, px_middle, px_end], [wave_start, wave_middle, wave_end], 2)
            return poly_coefs
    
    # Calculates the wavelengths of the pixels on a given order with the coefficients (no matter if linear or quadratic).
    def calc_order_wavelengths(self, order_idx):
        if order_idx > len(self.calib_data_dynamic):
            raise Exception(f'calc_order_wavelengths() | order_idx out of bounds: {order_idx}')
        
        x_array = self.orders_x_pixels[order_idx]
        coefs = self.orders_wavelength_coefficients[order_idx]
        wavelengths = np.polynomial.polynomial.polyval(x_array, coefs)
        return wavelengths
    
    
    # Create one array of wavelengths from all orders
    def compile_wavelengths(self):
        # Filter orders where use_order is True
        filtered_wavelengths = [wl_arr for idx, wl_arr in enumerate(self.orders_wavelengths) if self.get_static_data(idx).use_order]
        
        self.compiled_wavelengths = np.concatenate(filtered_wavelengths[::-1]) # reverse order (low nr is high wl)
    
    
    # Get polynomial coefficients of the calibration order curves
    def update_order_poly_coefs(self):
        
        for order_idx in range(len(self.calib_data_dynamic)):
            order = self.calib_data_dynamic[order_idx]
            poly_coefs = np.polynomial.polynomial.polyfit(order.xlist, order.ylist, 2)
            self.order_poly_coefs.append(poly_coefs)
    
    
    # Load correction multipliers from file if it exists, otherwise return None
    def get_correction_multipliers(self):
        filenames = get_folder_files_pattern(working_path, pattern = r'^_Correction_multipliers.*\.csv$')
        
        if len(filenames) > 1:
            print('Warning! More than one .csv multipliers file detected.')
        
        try:
            filepath = working_path + filenames[0]
            multipliers_array = np.loadtxt(filepath, delimiter = ',', skiprows = 1)
        except:
            print('_Correction_multipliers*.csv not found')
            self.multipliers = None
            return
        
        # Interpolate the multipliers to match the wavelengths of output spectra
        wavelengths_original = multipliers_array[:, 0]
        multipliers_original = multipliers_array[:, 1]
        
        # Create linear interpolator with extrapolation enabled
        linear_interp = interp1d(wavelengths_original, multipliers_original,
            kind='quadratic', fill_value='extrapolate', assume_sorted=False)
        
        # Interpolate multipliers at the spectrum wavelengths
        interpolated_multipliers = linear_interp(self.compiled_wavelengths)
        self.multipliers = interpolated_multipliers
    
    
    
    
    # Create output folder if it doesn't exist
    def create_folder(self):
        
        # Get subfolder name depending on whether correction multipliers file exists
        folder_name = 'Raw_spectra' + system_path_symbol
        if not self.multipliers is None:
            folder_name = 'Corrected_spectra' + system_path_symbol
        
        path = working_path + folder_name
        if not os.path.isdir(path):
            os.mkdir(path)
    
    
    
    ##############################################################################################################
    # Processing phase
    ##############################################################################################################
    
    def process_photo(self, filename):
        filepath = working_path + filename
        self.photo_array = self.load_photo(filepath)
        
        # pre-calculate cumulative column sums for integral calculation later
        self.update_cumulsum()
        
        # Get spectrum y values (z on image)
        self.update_spectrum_intensities()
        
        # Combine all orders and create one array for intensities
        self.compile_intensities()
        
        # Use spectral sensitivity multipliers
        self.apply_correction()
        
        
        # save the spectrum into txt file with ; delimiter (same as Sophi nXt)
        self.save_spectrum(filename)
    
    
    # Calculate cumulative sum for each column separately for faster integral calculation for spectrum
    # Gemini 2.5 Pro invention (and a good one)
    def update_cumulsum(self):
        """Pre-calculates and caches the cumulative sum of photo_array along axis 0."""
        # Pad with a row of zeros at the top for easier cumulsum indexing:
        # sum(S to E inclusive) = cumulsum[E+1] - cumulsum[S]
        cumsum_temp = np.cumsum(self.photo_array, axis=0)
        self.cumulative_sum = np.pad(cumsum_temp, ((1, 0), (0, 0)), 
                                                mode='constant', constant_values=0.)
        
    
    
    # Get intensities (integral over few px vertically) corresponding to the x_pixels
    def update_spectrum_intensities(self):
        for order_idx in range(len(self.calib_data_dynamic)):
            static_data = self.get_static_data(order_idx)
            
            # Ignore if use_order == False
            if static_data.use_order:
                self.update_spectrum_data_list(order_idx, self.orders_intensities, self.calc_order_intensities)
            else:
                self.orders_intensities.append(np.empty(0))
            
            
            
            
    
    # Calculates the wavelengths of the pixels on a given order with the coefficients (no matter if linear or quadratic).
    def calc_order_intensities(self, order_idx):
        if order_idx > len(self.calib_data_dynamic):
            raise Exception(f'calc_order_intensities() | order_idx out of bounds: {order_idx}')
        
        # Get radius over which to integrate given order
        radius = self.get_order_radius(order_idx)
        
        # Get coordinates of the center pixels 
        x_array = self.orders_x_pixels[order_idx]
        y_array = np.polynomial.polynomial.polyval(x_array, self.order_poly_coefs[order_idx])
        
        # Get z values from Echellogram
        integrals = self.integrate_order_width(x_array, y_array, radius = radius)        
        return integrals
    
    # Get the width to integrate over (between two diffraction orders)
    # Lower order nr has tighter order steps, so prefer lower index.
    def get_order_radius(self, order_idx):
        orders_dist = 1 # related to integral width (in Sophi settings) in some way
        center = self.calib_data_dynamic[order_idx].avg_y # Assumes that the orders are regular and don't have big changes between orders
        
        if order_idx <= 0: # check for out of bounds
            high = self.calib_data_dynamic[order_idx + 1].avg_y
            orders_dist = abs(high - center)
        
        else: # default
            low = self.calib_data_dynamic[order_idx - 1].avg_y
            orders_dist = abs(center - low)    
        
        # Integrate until 40 % to the next order to avoid overlap
        radius = orders_dist * 0.4
        
        # very important if radius is integer (not anymore), otherwise 0.49 and 0.51 will have 150% jump in integral because of rounding
        radius = clip(radius, min_v = 0.1)
        
        return radius
    
    
    # Sum pixels around the order with the given radius. The radius can be a float, only integers are x_array.
    # if use_weights is True then the integral is summed with Gaussian weights (max is in center) with FWHM of width
    def integrate_order_width(self, x_array, y_array, radius = 0, use_weights = False):
        
        # Clip coordinates to be within bounds of the image
        x_array = np.clip(x_array, 0, self.max_x_idx).astype(int) 
        y_array = np.clip(y_array, 0, self.max_y_idx).astype(float)
        
        # Return the intensities from the diffr. order curve
        if radius == 0:
            return self.photo_array[y_array, x_array] # x and y have to be switched (first dimension is vertical on image)
        
        integrals = self.integrate_order_width_interpolated(x_array, y_array, radius)
        return integrals
    
    
    # Made with Gemini 2.5 Pro (based on my algorithm)
    def integrate_order_width_interpolated(self, x_indices, y_indices_center_float, radius_float):
        """
        Integrates pixel intensities along y-columns with float bounds and quadratic interpolation.

        Args:
            x_indices (np.ndarray): 1D array of integer x-coordinates (columns).
            y_indices_center_float (np.ndarray): 1D array of float y-coordinates (centers of integration).
            radius_float (float or np.ndarray): Integration radius. Can be scalar or array.

        Returns:
            np.ndarray: 1D array of integrated intensity values.
        """
        
        
        # Ensure inputs are numpy arrays
        if not isinstance(x_indices, np.ndarray): 
            x_indices = np.array(x_indices, dtype=int)
        if not isinstance(y_indices_center_float, np.ndarray): 
            y_indices_center_float = np.array(y_indices_center_float, dtype=float)
        
        if np.isscalar(radius_float):
            radius_float = np.full_like(y_indices_center_float, float(radius_float), dtype=float)
        elif not isinstance(radius_float, np.ndarray):
            radius_float = np.array(radius_float, dtype=float)

        num_points = len(x_indices)
        if num_points == 0:
            return np.array([], dtype=float)
        
        # 1. Clip x_coords to be within image bounds (should be done by caller ideally)
        x_indices = np.clip(x_indices, 0, self.max_x_idx)
        
        # 2. Calculate float y-integration-bounds and clip them to image limits
        y_bounds_low_float = y_indices_center_float - radius_float
        y_bounds_high_float = y_indices_center_float + radius_float

        y_bounds_low_float = np.clip(y_bounds_low_float, 0.0, float(self.max_y_idx))
        y_bounds_high_float = np.clip(y_bounds_high_float, 0.0, float(self.max_y_idx))
        
        # 3. Determine integer y-indices for the "bulk" part of the sum
        # These define the inclusive range [y_bulk_sum_low_int, y_bulk_sum_high_int]
        y_bulk_sum_low_int = np.ceil(y_bounds_low_float).astype(int)
        y_bulk_sum_high_int = np.floor(y_bounds_high_float).astype(int)
        
        # Clip again to be absolutely sure after ceil/floor (mostly for type consistency)
        y_bulk_sum_low_int = np.clip(y_bulk_sum_low_int, 0, self.max_y_idx)
        y_bulk_sum_high_int = np.clip(y_bulk_sum_high_int, 0, self.max_y_idx)
        
        # 4. Calculate fractional lengths for the interpolated parts
        # low_fraction: portion from y_bounds_low_float up to the first full pixel
        low_fraction = np.abs(y_bulk_sum_low_int - y_bounds_low_float)
        # high_fraction: portion from the last full pixel up to y_bounds_high_float
        high_fraction = np.abs(y_bounds_high_float - y_bulk_sum_high_int)
        

        # 5. Calculate bulk sum using the pre-calculated cumulative sum array
        # Sum from S=y_bulk_sum_low_int to E=y_bulk_sum_high_int (inclusive) is:
        # cumulative_sum[E+1] - cumulative_sum[S]
        
        idx_E_plus_1 = y_bulk_sum_high_int + 1 # Upper index for cumulsum (exclusive in slicing terms)
        idx_S = y_bulk_sum_low_int            # Lower index for cumulsum (inclusive)
        
        term_high = self.cumulative_sum[idx_E_plus_1, x_indices]
        term_low = self.cumulative_sum[idx_S, x_indices]
        bulk_sum_values = term_high - term_low
        
        # Correct for cases where the bulk interval is empty (e.g., y_bulk_sum_high_int < y_bulk_sum_low_int)
        #empty_interval_mask = y_bulk_sum_high_int < y_bulk_sum_low_int
        #bulk_sum_values[empty_interval_mask] = 0.0
        
        # Initialize interpolated values
        interp_contrib_low = np.zeros(num_points, dtype=float)
        interp_contrib_high = np.zeros(num_points, dtype=float)

        # 6. Quadratic interpolation for the lower fractional part (only if low_fraction > 0)
        # Points where low_fraction is significant enough to warrant interpolation
        interp_contrib_low = self.edge_quadratic_interpolation(x_indices, low_fraction, y_bounds_low_float, interp_contrib_low)
        
        # 7. Quadratic interpolation for the upper fractional part (only if high_fraction > 0)
        interp_contrib_high = self.edge_quadratic_interpolation(x_indices, high_fraction, y_bounds_high_float, interp_contrib_high)
        
        # 8. Combine all parts: Bulk sum + contribution from lower fraction + contribution from upper fraction
        total_integrals = bulk_sum_values + interp_contrib_low + interp_contrib_high
        
        return total_integrals
    
    
    # Get the edge index (float fraction) value by quadratic interpolation
    def edge_quadratic_interpolation(self, x_indices, fraction, y_bounds_float, interp_contrib):
        needs_interp_mask = fraction > 1e-3 # Avoid calculations for zero fractions

        if np.any(needs_interp_mask):
            # Filter data for points needing low interpolation
            ylf_clip_subset = y_bounds_float[needs_interp_mask]
            x_coords_subset = x_indices[needs_interp_mask]
    
            # Determine stencil center: integer y pixel "closest" to the float boundary
            # Clip stencil center so that stencil_center +/- 1 are valid indices
            y_center_stencil = np.round(ylf_clip_subset).astype(int)
            y_center_stencil = np.clip(y_center_stencil, 1, self.max_y_idx - 1)
            
            yl_m1 = y_center_stencil - 1
            yl_0  = y_center_stencil
            yl_p1 = y_center_stencil + 1
            
            # Fetch z-values for the stencil points
            zl_m1 = self.photo_array[yl_m1, x_coords_subset]
            zl_0  = self.photo_array[yl_0,  x_coords_subset]
            zl_p1 = self.photo_array[yl_p1, x_coords_subset]
            
            val_at_bound = self.vectorized_quadratic_interpolate(ylf_clip_subset, y_center_stencil, zl_m1, zl_0, zl_p1)
            interp_contrib[needs_interp_mask] = val_at_bound * fraction[needs_interp_mask]
            
        return interp_contrib
    
    
    # Gemini 2.5 Pro invention for faster quadratic interpolation
    # Much faster than my simple_quadratic_interpolate() because np.polynomial.polynomial.polyfit() isn't easily vectorizable.
    # There is max about 20 % relative int difference between this and simple_quadratic_interpolate() integrals but it's mostly
    # for noise and compared to max intensity in the spectrum, completely negligible.
    def vectorized_quadratic_interpolate(self, y_target_float, y_center_int_stencil, 
                                          z_stencil_minus_1, z_stencil_0, z_stencil_plus_1):
        """
        Performs quadratic interpolation using a 3-point stencil.
        y_target_float: Array of N target y-coordinates (float).
        y_center_int_stencil: Array of N integer y-coordinates, center of the stencil [-1, 0, +1].
        z_stencil_minus_1, z_stencil_0, z_stencil_plus_1: Pixel values at stencil points.
        """
        y_rel = y_target_float - y_center_int_stencil # Target relative to stencil center
        
        # Lagrange coefficients for stencil points [-1, 0, 1] relative to y_center_int_stencil
        # L_minus_1 corresponds to z_stencil_minus_1 (at relative coordinate -1)
        # L_0 corresponds to z_stencil_0 (at relative coordinate 0)
        # L_plus_1 corresponds to z_stencil_plus_1 (at relative coordinate +1)
        
        L_minus_1 = y_rel * (y_rel - 1.0) / 2.0
        L_0 = 1.0 - y_rel**2  # Simplified from -(y_rel + 1.0) * (y_rel - 1.0)
        L_plus_1 = y_rel * (y_rel + 1.0) / 2.0
        
        return L_minus_1 * z_stencil_minus_1 + L_0 * z_stencil_0 + L_plus_1 * z_stencil_plus_1
    
    
    
    # Get z-values of corresponding x and y values and cut them according to left and right bounds
    def compile_intensities(self):
        # Filter orders where use_order is True
        filtered_intensities = [int_arr for idx, int_arr in enumerate(self.orders_intensities) if self.get_static_data(idx).use_order]
        
        self.compiled_intensities = np.concatenate(filtered_intensities[::-1]) # reverse order
    
    
    # Apply spectral sensitivity multipliers
    def apply_correction(self):
        if self.multipliers is None:
            return
        
        # Multiply arrays row by row
        self.compiled_intensities *= self.multipliers
    
    
    ##############################################################################################################
    # Output phase
    ##############################################################################################################
    
    # save the spectrum into txt file with ; delimiter (same as Sophi nXt)
    def save_spectrum(self, filename):
        
        # get filename without .tif
        pattern = r'(.+?)\.tif$'
        regex_result = re.search(pattern, filename)
        identificator = regex_result[1]
        
        # Get subfolder name depending on whether correction multipliers file exists
        folder_name = 'Raw_spectra' + system_path_symbol
        if not self.multipliers is None:
            folder_name = 'Corrected_spectra' + system_path_symbol
        
        filepath = working_path + folder_name + identificator + '.txt'
        
        output_array = np.column_stack((self.compiled_wavelengths, self.compiled_intensities))
        np.savetxt(filepath, output_array, delimiter = ';', fmt = "%.8f")


##############################################################################################################
# Utility functions
##############################################################################################################


def clip(value, min_v = -math.inf, max_v = math.inf):
    return max(min_v, min(value, max_v))

def get_folder_files_pattern(path, pattern = None, return_all = False):
    
    # Get files, not directories
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    onlyfiles.sort() # sort filenames
    
    # Extract all files
    if return_all or (pattern is None):
        return onlyfiles
    
    output_files = [f for f in onlyfiles if re.search(pattern, f)]
    return output_files

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