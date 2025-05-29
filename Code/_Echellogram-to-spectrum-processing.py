# Author: Jasper Ristkok
# v3.0

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
    def __init__(self, x, y = None):
        
        if x is None:
            raise ValueError('No x data')
        
        self.x = x
        self.y = y


# Line of a diffraction order (horizontal), x is horizontal and y vertical pixels
class dynamic_calibration_data():
    def __init__(self, image_width = 1, existing_data = None):
        self.image_width = image_width
        self.avg_y = 0
        self.xlist = None
        self.ylist = None
        self.points = []
        self.use_order = True
        
        self.load_encode(existing_data)
        self.update()
    
    
    # Convert saved dictionary into point classes
    def load_encode(self, existing_data):
        self.use_order = existing_data['use_order']
        self.points = []
        self.add_points(existing_data['x'], existing_data['y'])
        
        self.update()
    
    # Add a list of points
    def add_points(self, x_list, y_list):
        for idx, x in enumerate(x_list):
            y = y_list[idx]
            self.add_point(x, y, to_update = False)
        self.update()
    
    # Create a point instance
    def add_point(self, x, y, to_update = True):
        self.points.append(point_class(x, y))
        if to_update:
            self.update()
    
    # Create x and y coordinate lists of point instances
    def update(self):
        self.sort_by_x()
        self.create_lists()
        self.avg_y = np.average(self.ylist)
    
    # Sort points by ascending x value
    def sort_by_x(self):
        x_values = [obj.x for obj in self.points]
        sorted_idx = np.argsort(x_values)
        self.points = [self.points[idx] for idx in sorted_idx]
        
    # These are already sorted in self.sort_by_x()
    def create_lists(self):
        self.xlist = self.to_list(self.points, 'x')
        self.ylist = self.to_list(self.points, 'y')
    
    # Take axis string as argument and return a list of the axis values of the points
    def to_list(self, points, axis):
        array = []
        for point in self.points:
            array.append(getattr(point, axis, None))
        return array
    
    

# Class for sharing variables between functions and easier copy-paste from Calibrator code
class processor_class():
    
    def __init__(self):
        self.init_class_variables()
        self.main_code()
    
    def init_class_variables(self):
        self.working_path = working_path
        
        self.shift_wavelengths = True # This is a variable in Calibrator but a constant here
        
        self.first_order_nr = None
        self.origin_shift_right = None
        self.total_shift_right = None
        
        self.bounds_px = None
        
        self.photo_array = None
        self.cumulative_sum = None # for faster vertical integral calculation
        self.order_poly_coefs = []
        self.max_x_idx = None
        self.max_y_idx = None
        
        # Spectrum data
        self.orders_x_pixels = []
        self.orders_wavelengths = []
        self.orders_intensities = []
        self.compiled_wavelengths = None
        self.compiled_intensities = None
        self.multipliers = None
        
        # list to store calibration data diffraction order class instances
        # holds points that make up the drawn curves, the order of point classes can change during calibration
        self.calib_data_dynamic = None
    
    # Reset stuff after every used sample
    def sample_reset(self):
        self.orders_intensities = []
    
    def main_code(self):
        
        #################################
        # Preparation phase
        #################################
        
        self.load_calibration_data()
        if len(self.calib_data_dynamic) == 0:
            raise Exception('No calibration data!')
        
        
        filenames = self.get_photos_in_folder(working_path)
        
        # Get the array size of echellograms
        self.check_tif_array_size(filenames)
        
        # Get the output spectrum wavelengths (used here for interpolation)
        self.update_spectrum_data_wavelengths()
        
        # Create one array of wavelengths from all orders
        self.compile_wavelengths()
        
        self.update_order_poly_coefs()
        
        # If folder contains spectral sensitivity multipliers then it's written into this variable
        #self.get_correction_multipliers() TODO: check interpolation
        
        # Create folder for output, name depends on whether correction file exists
        self.create_folder()
        
        
        #################################
        # Processing phase
        #################################
        
        count = 0
        for filename in filenames:
            self.sample_reset()
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
    
    # Load data from Calibrator's .json file
    # Up shift info is irrelevant, since the dynamic list already contain shifted values
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
        
        
        # Load general info
        self.first_order_nr = calibration_data['first_order_nr']
        self.origin_shift_right = calibration_data['origin_shift_right']
        self.total_shift_right = calibration_data['total_shift_right']
        
        # Encode dictionary into data classes
        self.load_dynamic_data(calibration_data)
        self.update_bounds_data()
        
    
    def load_dynamic_data(self, calibration_data):
        self.calib_data_dynamic = []
        
        # Iterate over orders
        for order_raw in calibration_data['dynamic']:
            self.calib_data_dynamic.append(dynamic_calibration_data(existing_data = order_raw))
        
        # sort orders by average y values
        self.sort_orders()
    
    # Assumes that each order instance has been updated (avg_y is correct)
    def sort_orders(self):
        
        # Get indices to sort by
        y_values = [obj.avg_y for obj in self.calib_data_dynamic]
        sorted_idx = np.argsort(y_values) # top curves are low order nrs
        
        # Sort orders
        self.calib_data_dynamic = [self.calib_data_dynamic[idx] for idx in sorted_idx]
        
        # If initialized
        if len(self.order_poly_coefs) > 0:
            self.order_poly_coefs = [self.order_poly_coefs[idx] for idx in sorted_idx]
        
    
    # Get the order bounds with the corresponding shift
    # Sophi has order bounds (px) the same no matter how far right the actual image has shifted. 
    # I have doubts about that they did the proccess correctly, if Sophi tooltips show the actual data for the given order.
    def update_bounds_data(self, idx = None):
        if idx is None:
            self.bounds_px = bounds_data[:, [1,2]] + self.origin_shift_right + self.total_shift_right
        else:
            self.bounds_px[idx] = bounds_data[idx, [1,2]] + self.origin_shift_right + self.total_shift_right
    
    # Count the given order_idx in bounds data and apply the offset 
    # if bounds data starts at self.first_order_nr then offset is 0
    def get_raw_bounds_data_idx(self, order_idx):
        min_static_order_nr = int(bounds_data[0,0])
        offset = self.first_order_nr - min_static_order_nr
        bounds_data_idx = order_idx + offset
        return bounds_data_idx
    
    # Get static_idx corresponding to order_idx (avoid out of bounds)
    def get_bounds_data_idx(self, order_idx):
        bounds_data_idx = self.get_raw_bounds_data_idx(order_idx)
        
        # Clip to avoid going out of bounds, effectively extrapolating (copying) the bounds
        bounds_data_idx = clip(bounds_data_idx, min_v = 0, max_v = bounds_data.shape[0] - 1)
        return bounds_data_idx
    
    # Get the order nr in bounds_data corresponding to the order_idx drawn order
    def get_bounds_data_order_nr(self, order_idx):
        bounds_data_idx = self.get_bounds_data_idx(order_idx)
        return bounds_data[bounds_data_idx, 0]
    
    def get_order_nr_from_idx(self, order_idx):
        return self.first_order_nr + order_idx
    
    
    # Get all .tif files in folder
    def get_photos_in_folder(self, path):
        
        filenames = [f for f in os.listdir(working_path) if os.path.isfile(os.path.join(working_path, f))]
        if len(filenames) == 0:
            raise Exception('Error: no files in input folder: ' + str(working_path))
        
        pattern = '\.tif$'
        filenames = [f for f in filenames if re.search(pattern, f)]
        
        return filenames

    
    # Read a random tif file and check the array size of the echellograms
    def check_tif_array_size(self, filenames):
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
            
            # Ignore if use_order == False
            if self.calib_data_dynamic[order_idx].use_order:
                self.update_order_spectrum_data_wavelengths(order_idx)
            
            # Write empty arrays to keep indexing
            else:    
                order_nr = self.get_order_nr_from_idx(order_idx)
                print(f'Skipping order nr {order_nr}')
                self.orders_x_pixels.append(np.empty(0))
                self.orders_wavelengths.append(np.empty(0))
    
    # Update/initialize the data associated with the spectrum (wl and int etc.) for the given order
    def update_order_spectrum_data_wavelengths(self, order_idx):
        # Get x_pixel values between bounds (e.g. array of ints from 480 to 701)
        self.update_spectrum_data_list(order_idx, self.orders_x_pixels, self.calc_order_x_pixels)
        
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
        static_idx = self.get_bounds_data_idx(order_idx)
        [x_left, x_right] = self.bounds_px[static_idx]
        
        px_values = np.arange(x_left, x_right + 1, dtype = int)
        return px_values
    
    
    # Calculates the wavelengths of the pixels on a given order with the coefficients (no matter if linear or quadratic).
    def calc_order_wavelengths(self, order_idx):
        if order_idx > len(self.calib_data_dynamic):
            raise Exception(f'calc_order_wavelengths() | order_idx out of bounds: {order_idx}')
        
        order_nr = self.get_order_nr_from_idx(order_idx)
        x_array = self.orders_x_pixels[order_idx]
        
        shift_amount = 0
        if self.shift_wavelengths:
            shift_amount = self.origin_shift_right + self.total_shift_right
        
        wavelengths = order_px_to_wavelength(order_nr, x_array, horizontal_shift = shift_amount)
        return wavelengths
    
    
    # Create one array of wavelengths from all orders
    def compile_wavelengths(self):
        # Filter orders where use_order is True
        filtered_wavelengths = [wl_arr for idx, wl_arr in enumerate(self.orders_wavelengths) if self.calib_data_dynamic[idx].use_order]
        
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
        folder_name = '_Raw_spectra' + system_path_symbol
        if not self.multipliers is None:
            folder_name = '_Corrected_spectra' + system_path_symbol
        
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
        #self.apply_correction() # TODO: check if correct interpolation
        
        
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
            
            # Ignore if use_order == False
            if self.calib_data_dynamic[order_idx].use_order:
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
        filtered_intensities = [int_arr for idx, int_arr in enumerate(self.orders_intensities) if self.calib_data_dynamic[idx].use_order]
        
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
        folder_name = '_Raw_spectra' + system_path_symbol
        if not self.multipliers is None:
            folder_name = '_Corrected_spectra' + system_path_symbol
        
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
# Vertical bounds and wavelengths data
# These are copy-pasted from Calibrator code. Check that for more info.
##############################################################################################################


# The data is extracted from sample 492 (JET experiment). The data is constant throughout the experiments.
# Only the smallest order nr row is cut from e.g. 2024.08.12 calibration day spectra because the order was
# too close to the image edge.
# I don't use the wavelength data because the wavelength coefficients I calculated with 3 points are more accurate than the 2 points given here.
# spectrum start px (unsorted),spectrum end px (unsorted),order nr,image start px,image end px,nothing,start wavelength (nm),end wavelength (nm)
extracted_orders_info = np.array([
    [3.93010e+04,4.03060e+04,3.50000e+01,1.80000e+01,1.02300e+03,0.00000e+00,7.44703e+02,7.70245e+02],
    [3.84620e+04,3.93000e+04,3.60000e+01,1.40000e+01,8.52000e+02,0.00000e+00,7.23898e+02,7.44688e+02],
    [3.76480e+04,3.84610e+04,3.70000e+01,1.10000e+01,8.24000e+02,0.00000e+00,7.04251e+02,7.23884e+02],
    [3.68610e+04,3.76470e+04,3.80000e+01,1.30000e+01,7.99000e+02,0.00000e+00,6.85764e+02,7.04251e+02],
    [3.61020e+04,3.68600e+04,3.90000e+01,2.10000e+01,7.79000e+02,0.00000e+00,6.68368e+02,6.85742e+02],
    [3.55460e+04,3.61010e+04,4.00000e+01,2.13000e+02,7.68000e+02,0.00000e+00,6.56016e+02,6.68358e+02],
    [3.46510e+04,3.55450e+04,4.10000e+01,6.00000e+01,9.54000e+02,0.00000e+00,6.36635e+02,6.56012e+02],
    [3.39480e+04,3.46500e+04,4.20000e+01,7.00000e+01,7.72000e+02,0.00000e+00,6.21695e+02,6.36618e+02],
    [3.32820e+04,3.39470e+04,4.30000e+01,1.01000e+02,7.66000e+02,0.00000e+00,6.07893e+02,6.21693e+02],
    [3.26010e+04,3.32810e+04,4.40000e+01,1.02000e+02,7.82000e+02,0.00000e+00,5.94098e+02,6.07883e+02],
    [3.19800e+04,3.26000e+04,4.50000e+01,1.47000e+02,7.67000e+02,0.00000e+00,5.81803e+02,5.94083e+02],
    [3.12880e+04,3.19790e+04,4.60000e+01,1.09000e+02,8.00000e+02,0.00000e+00,5.68405e+02,5.81796e+02],
    [3.06560e+04,3.12870e+04,4.70000e+01,1.14000e+02,7.45000e+02,0.00000e+00,5.56406e+02,5.68391e+02],
    [3.00590e+04,3.06550e+04,4.80000e+01,1.41000e+02,7.37000e+02,0.00000e+00,5.45322e+02,5.56401e+02],
    [2.94670e+04,3.00580e+04,4.90000e+01,1.61000e+02,7.52000e+02,0.00000e+00,5.34560e+02,5.45312e+02],
    [2.88720e+04,2.94660e+04,5.00000e+01,1.66000e+02,7.60000e+02,0.00000e+00,5.23956e+02,5.34544e+02],
    [2.83010e+04,2.88710e+04,5.10000e+01,1.83000e+02,7.53000e+02,0.00000e+00,5.13981e+02,5.23939e+02],
    [2.77320e+04,2.83000e+04,5.20000e+01,1.92000e+02,7.60000e+02,0.00000e+00,5.04250e+02,5.13978e+02],
    [2.71860e+04,2.77310e+04,5.30000e+01,2.13000e+02,7.58000e+02,0.00000e+00,4.95090e+02,5.04244e+02],
    [2.66410e+04,2.71850e+04,5.40000e+01,2.25000e+02,7.69000e+02,0.00000e+00,4.86119e+02,4.95081e+02],
    [2.60540e+04,2.66400e+04,5.50000e+01,1.85000e+02,7.71000e+02,0.00000e+00,4.76619e+02,4.86108e+02],
    [2.55350e+04,2.60530e+04,5.60000e+01,2.01000e+02,7.19000e+02,0.00000e+00,4.68363e+02,4.76609e+02],
    [2.50200e+04,2.55340e+04,5.70000e+01,2.12000e+02,7.26000e+02,0.00000e+00,4.60317e+02,4.68352e+02],
    [2.44880e+04,2.50190e+04,5.80000e+01,1.97000e+02,7.28000e+02,0.00000e+00,4.52143e+02,4.60303e+02],
    [2.39870e+04,2.44870e+04,5.90000e+01,2.04000e+02,7.04000e+02,0.00000e+00,4.44584e+02,4.52141e+02],
    [2.35030e+04,2.39860e+04,6.00000e+01,2.19000e+02,7.02000e+02,0.00000e+00,4.37396e+02,4.44572e+02],
    [2.30130e+04,2.35020e+04,6.10000e+01,2.21000e+02,7.10000e+02,0.00000e+00,4.30253e+02,4.37396e+02],
    [2.25440e+04,2.30120e+04,6.20000e+01,2.35000e+02,7.03000e+02,0.00000e+00,4.23514e+02,4.30239e+02],
    [2.20560e+04,2.25430e+04,6.30000e+01,2.23000e+02,7.10000e+02,0.00000e+00,4.16617e+02,4.23505e+02],
    [2.15890e+04,2.20550e+04,6.40000e+01,2.24000e+02,6.90000e+02,0.00000e+00,4.10118e+02,4.16610e+02],
    [2.11360e+04,2.15880e+04,6.50000e+01,2.32000e+02,6.84000e+02,0.00000e+00,4.03917e+02,4.10117e+02],
    [2.06900e+04,2.11350e+04,6.60000e+01,2.40000e+02,6.85000e+02,0.00000e+00,3.97904e+02,4.03913e+02],
    [2.02490e+04,2.06890e+04,6.70000e+01,2.46000e+02,6.86000e+02,0.00000e+00,3.92043e+02,3.97895e+02],
    [1.98070e+04,2.02480e+04,6.80000e+01,2.45000e+02,6.86000e+02,0.00000e+00,3.86262e+02,3.92042e+02],
    [1.93690e+04,1.98060e+04,6.90000e+01,2.41000e+02,6.78000e+02,0.00000e+00,3.80610e+02,3.86255e+02],
    [1.89470e+04,1.93680e+04,7.00000e+01,2.47000e+02,6.68000e+02,0.00000e+00,3.75248e+02,3.80609e+02],
    [1.85370e+04,1.89460e+04,7.10000e+01,2.59000e+02,6.68000e+02,0.00000e+00,3.70112e+02,3.75246e+02],
    [1.81210e+04,1.85360e+04,7.20000e+01,2.59000e+02,6.74000e+02,0.00000e+00,3.64970e+02,3.70106e+02],
    [1.77130e+04,1.81200e+04,7.30000e+01,2.61000e+02,6.68000e+02,0.00000e+00,3.59993e+02,3.64961e+02],
    [1.73080e+04,1.77120e+04,7.40000e+01,2.61000e+02,6.65000e+02,0.00000e+00,3.55126e+02,3.59991e+02],
    [1.69200e+04,1.73070e+04,7.50000e+01,2.72000e+02,6.59000e+02,0.00000e+00,3.50520e+02,3.55119e+02],
    [1.65280e+04,1.69190e+04,7.60000e+01,2.74000e+02,6.65000e+02,0.00000e+00,3.45930e+02,3.50514e+02],
    [1.61440e+04,1.65270e+04,7.70000e+01,2.79000e+02,6.62000e+02,0.00000e+00,3.41493e+02,3.45925e+02],
    [1.57620e+04,1.61430e+04,7.80000e+01,2.81000e+02,6.62000e+02,0.00000e+00,3.37136e+02,3.41488e+02],
    [1.53940e+04,1.57610e+04,7.90000e+01,2.92000e+02,6.59000e+02,0.00000e+00,3.32991e+02,3.37129e+02],
    [1.50140e+04,1.53930e+04,8.00000e+01,2.87000e+02,6.66000e+02,0.00000e+00,3.28770e+02,3.32990e+02],
    [1.46500e+04,1.50130e+04,8.10000e+01,2.93000e+02,6.56000e+02,0.00000e+00,3.24775e+02,3.28767e+02],
    [1.43010e+04,1.46490e+04,8.20000e+01,3.09000e+02,6.57000e+02,0.00000e+00,3.20987e+02,3.24766e+02],
    [1.39330e+04,1.43000e+04,8.30000e+01,3.02000e+02,6.69000e+02,0.00000e+00,3.17042e+02,3.20978e+02],
    [1.35760e+04,1.39320e+04,8.40000e+01,3.02000e+02,6.58000e+02,0.00000e+00,3.13264e+02,3.17039e+02],
    [1.32330e+04,1.35750e+04,8.50000e+01,3.11000e+02,6.53000e+02,0.00000e+00,3.09671e+02,3.13254e+02],
    [1.28830e+04,1.32320e+04,8.60000e+01,3.10000e+02,6.59000e+02,0.00000e+00,3.06057e+02,3.09671e+02],
    [1.25520e+04,1.28820e+04,8.70000e+01,3.23000e+02,6.53000e+02,0.00000e+00,3.02670e+02,3.06047e+02],
    [1.22040e+04,1.25510e+04,8.80000e+01,3.16000e+02,6.63000e+02,0.00000e+00,2.99156e+02,3.02667e+02],
    [1.18750e+04,1.22030e+04,8.90000e+01,3.24000e+02,6.52000e+02,0.00000e+00,2.95872e+02,2.99154e+02],
    [1.15510e+04,1.18740e+04,9.00000e+01,3.33000e+02,6.56000e+02,0.00000e+00,2.92671e+02,2.95866e+02],
    [1.12200e+04,1.15500e+04,9.10000e+01,3.32000e+02,6.62000e+02,0.00000e+00,2.89441e+02,2.92669e+02],
    [1.08900e+04,1.12190e+04,9.20000e+01,3.28000e+02,6.57000e+02,0.00000e+00,2.86252e+02,2.89436e+02],
    [1.05720e+04,1.08890e+04,9.30000e+01,3.32000e+02,6.49000e+02,0.00000e+00,2.83209e+02,2.86244e+02],
    [1.02530e+04,1.05710e+04,9.40000e+01,3.32000e+02,6.50000e+02,0.00000e+00,2.80192e+02,2.83205e+02],
    [9.95300e+03,1.02520e+04,9.50000e+01,3.48000e+02,6.47000e+02,0.00000e+00,2.77390e+02,2.80192e+02],
    [9.64200e+03,9.95200e+03,9.60000e+01,3.50000e+02,6.60000e+02,0.00000e+00,2.74515e+02,2.77388e+02],
    [9.32900e+03,9.64100e+03,9.70000e+01,3.46000e+02,6.58000e+02,0.00000e+00,2.71643e+02,2.74506e+02],
    [9.02700e+03,9.32800e+03,9.80000e+01,3.50000e+02,6.51000e+02,0.00000e+00,2.68903e+02,2.71638e+02],
    [8.72300e+03,9.02600e+03,9.90000e+01,3.49000e+02,6.52000e+02,0.00000e+00,2.66173e+02,2.68898e+02],
    [8.42700e+03,8.72200e+03,1.00000e+02,3.53000e+02,6.48000e+02,0.00000e+00,2.63542e+02,2.66169e+02],
    [8.12800e+03,8.42600e+03,1.01000e+02,3.51000e+02,6.49000e+02,0.00000e+00,2.60910e+02,2.63538e+02],
    [7.83700e+03,8.12700e+03,1.02000e+02,3.54000e+02,6.44000e+02,0.00000e+00,2.58374e+02,2.60906e+02],
    [7.55100e+03,7.83600e+03,1.03000e+02,3.59000e+02,6.44000e+02,0.00000e+00,2.55903e+02,2.58368e+02],
    [7.26400e+03,7.55000e+03,1.04000e+02,3.60000e+02,6.46000e+02,0.00000e+00,2.53446e+02,2.55895e+02],
    [6.98200e+03,7.26300e+03,1.05000e+02,3.64000e+02,6.45000e+02,0.00000e+00,2.51061e+02,2.53444e+02],
    [6.70500e+03,6.98100e+03,1.06000e+02,3.70000e+02,6.46000e+02,0.00000e+00,2.48737e+02,2.51056e+02],
    [6.42900e+03,6.70400e+03,1.07000e+02,3.75000e+02,6.50000e+02,0.00000e+00,2.46449e+02,2.48737e+02],
    [6.14500e+03,6.42800e+03,1.08000e+02,3.69000e+02,6.52000e+02,0.00000e+00,2.44111e+02,2.46444e+02],
    [5.87100e+03,6.14400e+03,1.09000e+02,3.70000e+02,6.43000e+02,0.00000e+00,2.41874e+02,2.44105e+02],
    [5.59600e+03,5.87000e+03,1.10000e+02,3.68000e+02,6.42000e+02,0.00000e+00,2.39652e+02,2.41871e+02],
    [5.33100e+03,5.59500e+03,1.11000e+02,3.73000e+02,6.37000e+02,0.00000e+00,2.37527e+02,2.39646e+02],
    [5.07200e+03,5.33000e+03,1.12000e+02,3.82000e+02,6.40000e+02,0.00000e+00,2.35472e+02,2.37524e+02],
    [4.81200e+03,5.07100e+03,1.13000e+02,3.88000e+02,6.47000e+02,0.00000e+00,2.33429e+02,2.35470e+02],
    [4.54800e+03,4.81100e+03,1.14000e+02,3.87000e+02,6.50000e+02,0.00000e+00,2.31367e+02,2.33421e+02],
    [4.28700e+03,4.54700e+03,1.15000e+02,3.87000e+02,6.47000e+02,0.00000e+00,2.29348e+02,2.31362e+02],
    [4.03300e+03,4.28600e+03,1.16000e+02,3.92000e+02,6.45000e+02,0.00000e+00,2.27403e+02,2.29345e+02],
    [3.78200e+03,4.03200e+03,1.17000e+02,3.98000e+02,6.48000e+02,0.00000e+00,2.25498e+02,2.27401e+02],
    [3.52100e+03,3.78100e+03,1.18000e+02,3.92000e+02,6.52000e+02,0.00000e+00,2.23534e+02,2.25497e+02],
    [3.28400e+03,3.52000e+03,1.19000e+02,4.07000e+02,6.43000e+02,0.00000e+00,2.21761e+02,2.23528e+02],
    [3.02500e+03,3.28300e+03,1.20000e+02,3.99000e+02,6.57000e+02,0.00000e+00,2.19846e+02,2.21761e+02],
    [2.77700e+03,3.02400e+03,1.21000e+02,3.99000e+02,6.46000e+02,0.00000e+00,2.18022e+02,2.19840e+02],
    [2.53900e+03,2.77600e+03,1.22000e+02,4.07000e+02,6.44000e+02,0.00000e+00,2.16286e+02,2.18016e+02],
    [2.29000e+03,2.53800e+03,1.23000e+02,4.02000e+02,6.50000e+02,0.00000e+00,2.14483e+02,2.16280e+02],
    [2.05000e+03,2.28900e+03,1.24000e+02,4.04000e+02,6.43000e+02,0.00000e+00,2.12760e+02,2.14478e+02],
    [1.81300e+03,2.04900e+03,1.25000e+02,4.07000e+02,6.43000e+02,0.00000e+00,2.11072e+02,2.12754e+02],
    [1.57800e+03,1.81200e+03,1.26000e+02,4.10000e+02,6.44000e+02,0.00000e+00,2.09410e+02,2.11065e+02],
    [1.34900e+03,1.57700e+03,1.27000e+02,4.18000e+02,6.46000e+02,0.00000e+00,2.07809e+02,2.09409e+02],
    [1.09900e+03,1.34800e+03,1.28000e+02,4.03000e+02,6.52000e+02,0.00000e+00,2.06072e+02,2.07806e+02],
    [8.93000e+02,1.09800e+03,1.29000e+02,4.30000e+02,6.35000e+02,0.00000e+00,2.04654e+02,2.06070e+02],
    [7.02000e+02,8.92000e+02,1.30000e+02,4.71000e+02,6.61000e+02,0.00000e+00,2.03354e+02,2.04654e+02],
    [0.00000e+00,7.01000e+02,1.31000e+02,0.00000e+00,7.01000e+02,0.00000e+00,1.98536e+02,2.03353e+02]
])


# This array is gained by fitting quadratic fn to every order in Sophi nXt output spectrum for sample 492.
# Since the sample spectrum had shifted (misaligned) then self.origin_shift_right corrects the misalignment.
# self.origin_shift_right makes sure the spectral lines have correct wavelengths when using these coefficients.
# y = c + b * px + a * px^2
#order_nr, c, b, a 
wavelength_calculation_coefficients = np.array([
[35, 744.6606369, 0.026219046, -6.99723E-07],
[36, 723.9690378, 0.025487264, -6.77505E-07],
[37, 704.3971515, 0.024796671, -6.58831E-07],
[38, 685.8562886, 0.024142858, -6.41254E-07],
[39, 668.2669108, 0.02352298, -6.24737E-07],
[40, 651.556411, 0.022937973, -6.11995E-07],
[41, 635.6633543, 0.022376726, -5.9762E-07],
[42, 620.5265819, 0.021842352, -5.80982E-07],
[43, 606.093802, 0.021334739, -5.67896E-07],
[44, 592.3172827, 0.020849941, -5.55284E-07],
[45, 579.1528605, 0.020387273, -5.43408E-07],
[46, 566.5611796, 0.019943761, -5.31571E-07],
[47, 554.5051032, 0.019519183, -5.19657E-07],
[48, 542.951144, 0.019113054, -5.09113E-07],
[49, 531.8686875, 0.018723613, -4.99193E-07],
[50, 521.2294916, 0.018349491, -4.89398E-07],
[51, 511.0073567, 0.017990125, -4.7995E-07],
[52, 501.1782737, 0.017644591, -4.70933E-07],
[53, 491.7199239, 0.017312214, -4.62289E-07],
[54, 482.6117728, 0.016992135, -4.54008E-07],
[55, 473.8349314, 0.016682891, -4.45349E-07],
[56, 465.3712357, 0.016385065, -4.37024E-07],
[57, 457.2043406, 0.016098067, -4.29566E-07],
[58, 449.3190416, 0.015820591, -4.22036E-07],
[59, 441.7008594, 0.015552614, -4.14721E-07],
[60, 434.3364439, 0.015293827, -4.07953E-07],
[61, 427.2134031, 0.015043439, -4.01376E-07],
[62, 420.3199984, 0.014801163, -3.94982E-07],
[63, 413.645398, 0.01456637, -3.88673E-07],
[64, 407.1792584, 0.014338876, -3.82427E-07],
[65, 400.9119379, 0.014118553, -3.76571E-07],
[66, 394.8344317, 0.013904952, -3.70956E-07],
[67, 388.9382569, 0.013697703, -3.6549E-07],
[68, 383.2154329, 0.013496464, -3.60112E-07],
[69, 377.6584204, 0.013300971, -3.54793E-07],
[70, 372.2600699, 0.013111151, -3.49695E-07],
[71, 367.0136792, 0.0129268, -3.44878E-07],
[72, 361.9129747, 0.012747486, -3.40143E-07],
[73, 356.9519515, 0.012573024, -3.35455E-07],
[74, 352.124944, 0.012403274, -3.30901E-07],
[75, 347.4265643, 0.012238133, -3.26534E-07],
[76, 342.8517748, 0.012077325, -3.22305E-07],
[77, 338.3957532, 0.011920664, -3.18138E-07],
[78, 334.0539359, 0.011768007, -3.14078E-07],
[79, 329.8219633, 0.011619275, -3.10167E-07],
[80, 325.6957679, 0.011474175, -3.06308E-07],
[81, 321.6713946, 0.011332651, -3.02499E-07],
[82, 317.745092, 0.011194729, -2.98939E-07],
[83, 313.9133978, 0.011059994, -2.95376E-07],
[84, 310.1728992, 0.01092839, -2.91782E-07],
[85, 306.5203422, 0.010799989, -2.8838E-07],
[86, 302.9526999, 0.010674553, -2.85064E-07],
[87, 299.4670145, 0.010552044, -2.81839E-07],
[88, 296.0605411, 0.010432246, -2.78658E-07],
[89, 292.7305713, 0.01031514, -2.75508E-07],
[90, 289.4745483, 0.010200726, -2.72537E-07],
[91, 286.290074, 0.010088757, -2.69576E-07],
[92, 283.1748155, 0.009979136, -2.66588E-07],
[93, 280.1265078, 0.009871915, -2.63697E-07],
[94, 277.1430282, 0.009766993, -2.60899E-07],
[95, 274.2223017, 0.009664375, -2.58237E-07],
[96, 271.362409, 0.009563876, -2.55644E-07],
[97, 268.5614874, 0.009465323, -2.52971E-07],
[98, 265.8176921, 0.009368812, -2.50373E-07],
[99, 263.1293055, 0.009274255, -2.47844E-07],
[100, 260.4946596, 0.009181597, -2.45367E-07],
[101, 257.9121702, 0.009090756, -2.42932E-07],
[102, 255.3802939, 0.009001698, -2.40539E-07],
[103, 252.8975513, 0.008914407, -2.38234E-07],
[104, 250.4625402, 0.008828777, -2.35962E-07],
[105, 248.0738896, 0.008744781, -2.33733E-07],
[106, 245.7302843, 0.008662393, -2.31569E-07],
[107, 243.4304664, 0.008581552, -2.29456E-07],
[108, 241.1732443, 0.008502124, -2.27309E-07],
[109, 238.9574218, 0.008424144, -2.2518E-07],
[110, 236.7818691, 0.008347601, -2.23117E-07],
[111, 234.6454927, 0.008272461, -2.21108E-07],
[112, 232.5472364, 0.008198724, -2.19199E-07],
[113, 230.4861047, 0.008126294, -2.1733E-07],
[114, 228.461137, 0.008055069, -2.15434E-07],
[115, 226.4713778, 0.007985061, -2.13546E-07],
[116, 224.5159031, 0.007916294, -2.11721E-07],
[117, 222.5938366, 0.007848731, -2.09959E-07],
[118, 220.7043557, 0.007782245, -2.0817E-07],
[119, 218.8465979, 0.00771694, -2.06452E-07],
[120, 217.019807, 0.007652694, -2.04762E-07],
[121, 215.2232133, 0.00758944, -2.03014E-07],
[122, 213.4560398, 0.00752731, -2.01381E-07],
[123, 211.7176043, 0.007466152, -1.99749E-07],
[124, 210.0072009, 0.00740596, -1.98114E-07],
[125, 208.3241463, 0.007346768, -1.96544E-07],
[126, 206.6677975, 0.007288519, -1.95004E-07],
[127, 205.0375157, 0.007231218, -1.93517E-07],
[128, 203.4327313, 0.007174703, -1.91962E-07],
[129, 201.8527768, 0.007119194, -1.90522E-07],
[130, 200.2970552, 0.007064804, -1.89371E-07],
[131, 198.7659124, 0.007008419, -1.85906E-07]
])


# order_nr, start px, end px
bounds_data = extracted_orders_info[:, [2,3,4]]

# sort in ascending order nr
bounds_data = bounds_data[bounds_data[:, 0].argsort()]
wavelength_calculation_coefficients = wavelength_calculation_coefficients[wavelength_calculation_coefficients[:, 0].argsort()]


# Calculate the wavelength of a pixel of a given order with given horizontal shift. px_array can be scalar or 1D array.
# Wavelengths of other spectra can be calculated with 1 pm accuracy with horizontal shift.
def order_px_to_wavelength(order_nr, px_array_orig, horizontal_shift = 0):
    order_nrs = wavelength_calculation_coefficients[:,0] #order_nr, c, b, a 
    first_order_in_data = int(bounds_data[0,0]) # already sorted
    last_order_in_data = int(bounds_data[-1,0]) # already sorted
    
    # Calculate coefficients from regression fromula if order nr is not defined in data
    if (order_nr < first_order_in_data) or (order_nr > last_order_in_data):
        
        # These values are gained by analyzing the spectrum from Sophi nXt with sample 492.
        # Each order and its wavelengths follow a quadratic fn y = a * x^2 + b * x + c
        # px in the equation is the image px, not order px (latter starting from left bound). 
        # px is from 0 to 1023 and bounds don't matter.
        # These coefficients in turn follow a very strong (smallest R^2 was 0.99997) linear (division) curve.
        a = -2.43686E-05 / order_nr - 1.68E-09
        b = 0.917000145 / order_nr + 1.11102E-05
        c = 26072.63016 / order_nr - 0.230703041
        coefs = np.array([c, b, a])
    
    
    # The direct polynomial coefficients still give more accurate result than a,b,c functions.
    # Get index of row where order_nr is the same and get the corresponding coefficients.
    else:
        idxs = np.where(order_nr == order_nrs)[0]
        if len(idxs) == 0:
            raise Exception(f'ERROR: order_px_to_wavelength() | no match for order {order_nr}. Check wavelength_calculation_coefficients for irregular order_nr-s.')
        else: 
            idx = idxs[0]
        coefs = wavelength_calculation_coefficients[idx, 1:4]
    
    px_array = px_array_orig.astype(float) # since shift is float but px is int
    px_array -= horizontal_shift
    return np.polynomial.polynomial.polyval(px_array, coefs)


##############################################################################################################
# RUN MAIN PROGRAM
##############################################################################################################
main_program()