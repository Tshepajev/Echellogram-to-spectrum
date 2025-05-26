# Author: Jasper Ristkok
# v2.3

# Code to convert an echellogram (photo) to spectrum


# TODO: check order edit mode
# TODO: better align (interpolate) calibration curves



##############################################################################################################
# CONSTANTS
##############################################################################################################

# If this is True then script assumes you're babysitting the code and are giving input
# Otherwise the code tries to do most stuff automatically like as with running the executable
# E.g. manual mode saves averaged spectrum for future use
script_manual_mode = False


# filename (for shot series) to use where index is _0001 with the file extension (.tif)
# '14IWG1A_11a_P11_gate_1000ns_delay_1000ns_0001.tif' '492_2X8_R7C3C7_0001 2X08 - R7 C3-C7.tif' 'Aryelle_0001.tif'
# 'Plansee_W_3us_gate_0001.tif' 'Plansee_W__0001.tif' 'IU667_D10_4us_0001.tif' 'IU667_D10_0001.tif' 
# 'integrating_sphere_100ms_10avg_fiber600umB_0001.tif' 
# 'DHLamp__0001.tif' 'Hg_lamp_0001.tif' 'W_Lamp_0001.tif' 'Ne_lamp_100ms_fiber600umB_0001.tif'
series_filename = None

# How many Echellograms to average starting from the first
# None means all in the shot series are used 
average_photos_nr = 20



# If working_path is None then the same folder will be used for all inputs and outputs where the code is executed
# Otherwise the defined custom paths are used that you define here
# If you aren't using Windows then paths should have / instead of \\ (double because \ is special character in strings)
working_path = None
#working_path = 'E:\\Nextcloud sync\\Data_processing\\Projects\\2024_09_JET\\Lab_comparison_test\\Photo_to_spectrum\\Photos_spectra_comparison\\'

system_path_symbol = '\\' # / for Unix, code checks for it later

# When using script in automatic mode then input folder is the same as output folder
# Then you can still use e.g. vertical points input but you have to put the file in the same
# folder as Echellograms and outputs
input_photos_path = None if working_path is None else working_path + 'Input_photos' + system_path_symbol
input_data_path = None if working_path is None else working_path + 'Input_data' + system_path_symbol
averaged_path = None if working_path is None else working_path + 'Averaged' + system_path_symbol
output_path = None if working_path is None else working_path + 'Output' + system_path_symbol
spectrum_path = None if working_path is None else working_path + 'Spectra' + system_path_symbol


# debug mode - prints amount of times each function is called
dbg = False


##############################################################################################################
# IMPORTS
##############################################################################################################

import numpy as np
import math
import os
import re # regex
import json
import tkinter # Tkinter
from tkinter import ttk

from PIL import Image as PILImage
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from  matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import platform

##############################################################################################################
# Debugging functions
##############################################################################################################

# Low priority, so taken from Chatgpt
if dbg:
    import functools
    import types
    function_call_nr = {}
    
    # Debugging wrapper to print function calls
    class FunctionLogger:
        def __init__(self, func):
            self.func = func
        
        def __call__(self, *args, **kwargs):
            #print(f"Calling function: {self.func.__name__}")
            global function_call_nr
            function_call_nr[self.func.__name__] += 1
            print(function_call_nr)
            return self.func(*args, **kwargs)
    
    def wrap_functions_in_module():
        global function_call_nr
        for name, obj in globals().items():
            if isinstance(obj, types.FunctionType):  # Check if it's a function
                function_call_nr[name] = 0
                globals()[name] = FunctionLogger(obj)  # Wrap the function
    
    
    def count_calls(cls):
        """
        Class decorator that injects call counters into **all**
        methods defined directly on the class.
        """
        
        for name, attr in cls.__dict__.items():
    
            # Skip non‑callables and dunder methods you don’t want to track
            if name.startswith("__") and name.endswith("__"):
                continue
    
            # We need to treat plain functions, staticmethods and classmethods separately
            original = (
                attr.__func__ if isinstance(attr, (staticmethod, classmethod)) else attr
            )
    
            if callable(original):
                function_call_nr[name] = 0
    
                @functools.wraps(original)
                def wrapper(*args, __orig=original, __name=name, **kwargs):
                    # __name and __orig captured as defaults so
                    # each wrapper keeps its own references
                    function_call_nr[__name] += 1
                    return __orig(*args, **kwargs)
                
                # Re‑wrap as staticmethod / classmethod if necessary
                if isinstance(attr, staticmethod):
                    wrapper = staticmethod(wrapper)
                elif isinstance(attr, classmethod):
                    wrapper = classmethod(wrapper)
    
                setattr(cls, name, wrapper)
    
        return cls



##############################################################################################################
# DEFINITIONS
# Definitions are mostly structured in the order in which they are called at first.
##############################################################################################################


def main_program():
    
    # Code is executed from folder with Echellograms
    if working_path is None:
        get_paths()
    
    # create folders if missing
    create_folder_structure()
    
    # Create and draw GUI window
    tkinter_master = tkinter.Tk()
    try:
        window = calibration_window(tkinter_master)
        
        # Set the window to be maximized
        if platform.system() == 'Windows':
            tkinter_master.state('zoomed')
        #else: # can't test because I don't have Mac/Linux
        #    tkinter_master.attributes('-zoomed', True)
        
        tkinter_master.mainloop() # run tkinter (only once)
    
    finally:
        
        # When code crashes or debugging is stopped then the tkinter object isn't destroyed in Spyder
        # This isn't probably an issue in the final program version.
        try: # another try in case user closed the window manually and tkinter_master doesn't exist
            tkinter_master.destroy()
        except:
            pass
    

##############################################################################################################
# Preparation phase
##############################################################################################################

# if program is in automatic mode (you don't execute it for a Python IDE and change the code here) then 
# use all input and output paths as the folder where the script (.py file) is executed from
def get_paths():
    
    # root path as the location where the python script was executed from 
    # (put the script .py file into the folder with the Echellograms)
    global system_path_symbol, working_path, input_photos_path, input_data_path, averaged_path, output_path, spectrum_path
    
    # Get OS name for determining path symbol
    if platform.system() == 'Windows':
        system_path_symbol = '\\'
    else:
        system_path_symbol = '/' # Mac, Linux
    
    # Get folder where .py file was executed from
    working_path = os.getcwd() # 'd:\\github\\echellogram-to-spectrum\\code'
    working_path += system_path_symbol
    
    input_photos_path = working_path
    input_data_path = working_path
    averaged_path = working_path
    output_path = working_path
    spectrum_path = working_path
    

def create_folder_structure():
    create_folder(output_path)
    create_folder(averaged_path)
    create_folder(input_photos_path)
    create_folder(input_data_path)
    create_folder(spectrum_path)
    
        
# returns array of tif image and filename where index is _0001 with the file extension (.tif)
def prepare_photos(series_filename = None):
    exif_data = None
    
    # Get files
    averaged_files = get_folder_files(averaged_path, return_all = True)
    
    if len(averaged_files) > 0:
        random_file = averaged_files[0] # take first file in folder, has _0001 due to sorting
    else:
        random_file = ''
    
    # Check if series_filename exists in files 
    if (series_filename is None) or (not series_filename in averaged_files):
        identificator = random_file
    else:
        identificator = series_filename
    
    # strip _0001 from the filename
    averaged_name = 'average_' + identificator.replace('_0001', '')
    
    # Load pre-existing averaged Echellogram
    if script_manual_mode and (averaged_name in averaged_files): # averaging done previously
        average_array, exif_data = load_photo(averaged_path + averaged_name)
    
    # Average first average_nr (relevant) photos in input folder and save averaged photo into averaged folder
    else:
        input_files = get_folder_files(input_photos_path, series_filename = identificator)
        
        if len(input_files) == 0:
            raise Exception('Error: no files in input folder: ' + str(input_photos_path))
        
        average_array, exif_data = average_photos(input_files, average_nr = average_photos_nr)
        
        if script_manual_mode:
            output_averaged_photo(average_array, output_path, averaged_name, exif_data)
    
    
    return average_array, identificator


def load_photo(filepath):
    image = PILImage.open(filepath)
    #im.show()
    
    #exif_data = image.info['exif'] # save exif data, so image can be opened with Sophi nXt
    exif_data = image.getexif() # save exif data, so image can be opened with Sophi nXt
    # TODO: get exif data (especially description (or comment) part) and save with new image
    
    imArray = np.array(image)
    return imArray, exif_data


def average_photos(input_files, average_nr = None):
    
    if average_nr is None:
        average_nr = len(input_files) # iterate over all files
    
    # Initialize averaging variables
    count = 0
    sum = None
    
        
    # iterate over files, files are already sorted in get_folder_files()
    for filename in input_files:
        imArray, exif_data = load_photo(input_photos_path + filename)
        
        # sum arrays
        if sum is None: # initialize
            sum = imArray
        else:  # add
            sum += imArray
        count += 1
        
        if count >= average_nr:
            break
    
    average = sum / count
    return average, exif_data


def output_averaged_photo(average_array, output_path, sample_name, exif_data = {}):
    #for k, v in exif_data.items():
    #    print("Tag", k, "Value", v)  # Tag 274 Value 2
    
    image = PILImage.fromarray(average_array)
    image.save(averaged_path + sample_name)#, exif = exif_data) 


def process_photo(photo_array):
    
    # negative values to 0
    photo_array[photo_array < 0] = 0
    
    # Load calibration data if it exists
    calibr_data, bounds_input_array = load_calibration_data()
    
    # Save averaged Echellogram as an array
    if script_manual_mode:
        np.savetxt(output_path + 'image_array.csv', photo_array, delimiter = ',')
    
    return photo_array, calibr_data, bounds_input_array

def load_calibration_data():
    calibration_data = None
    
    input_files = get_folder_files(input_data_path, return_all = True)
    
    # calibration done previously, load that data
    if '_Calibration_data.json' in input_files:  # _ in front to find easily among Echellograms
        with open(input_data_path + '_Calibration_data.json', 'r') as file:
            try:
                calibration_data = json.load(file)
            except: # e.g. file is empty
                pass
    
    bounds_input_array = None
    input_data_files = get_folder_files(input_data_path, return_all = True)
    if '_Vertical_points.csv' in input_data_files: # _ in front to find easily among Echellograms
        bounds_input_array = np.loadtxt(input_data_path + '_Vertical_points.csv', delimiter = ',', skiprows = 1)
    
    return calibration_data, bounds_input_array


###########################################################################################
# Point and diffraction order classes
###########################################################################################

# Generic point class, a point on a graph
class point_class():
    def __init__(self, x, y = None, z = None, group = None):
        
        if x is None:
            raise ValueError('No x data')
        
        self.x = x
        self.y = y
        self.z = z
        self.group = group
    
    def distance(self, point2):
        sum = (point2.x - self.x) ** 2
        
        if not (self.y is None):
            if point2.y is None:
                raise Exception('Dimensions of points don\'t match')
            sum += (point2.y - self.y) ** 2
        
        if not (self.z is None):
            if point2.z is None:
                raise Exception('Dimensions of points don\'t match')
            sum += (point2.z - self.z) ** 2
        
        return np.sqrt(sum)

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
        
        if existing_data:
            self.load_encode(existing_data)
    
    # Convert points into arrays for saving
    def save_decode(self):
        static_dict = {}
        static_dict['order_nr'] = self.order_nr
        static_dict['bounds_px'] = self.bounds_px
        static_dict['bounds_wave'] = self.bounds_wave
        static_dict['bounds_middle'] = self.bounds_middle
        static_dict['bounds_px_original'] = self.bounds_px_original
        static_dict['bounds_wave_original'] = self.bounds_wave_original
        static_dict['bounds_middle_orig'] = self.bounds_middle_orig
        static_dict['use_order'] = self.use_order
        
        return static_dict
    
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
        #self.xlist = None
        #self.ylist = None
        #self.points = []
        
        if existing_data is None:
            self.points = [point_class(0, 1, group = 'orders'), 
                        point_class(self.image_width / 2 - 1, 1, group = 'orders'), 
                        point_class(self.image_width - 1, 1, group = 'orders')]
            
        else:
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
    
    
    # Convert points into arrays for saving
    def save_decode(self):
        
        # Iterate over points and decode them
        points_array = []
        for point in self.points:
            point_dict = {}
            point_dict['x'] = point.x
            point_dict['y'] = point.y
            
            points_array.append(point_dict)
        
        return points_array
    
    # Convert saved dictionary into point classes
    def load_encode(self, existing_data):
        self.points = []
        
        # Iterate over points and encode them into classes
        for point_dict in existing_data:
            self.points.append(point_class(point_dict['x'], y = point_dict['y']))
        
        self.update()
    
    
###########################################################################################
# Main class
###########################################################################################

# Window to calibrate the echellogram to spectrum curves
class calibration_window():
    
    # Main code
    def __init__(self, parent):
        
        # Initialize argument variables
        self.root = parent
        
        # Initialize class variables
        self.init_class_variables()
        
        # Draw main window and tkinter elements
        self.initialize_window_elements()
        
        # Load data from the folder and initialize program with it
        self.load_folder(working_path)
        
        # Draw tkinter window elements (needs to be after drawing the plots)
        self.pack_window_elements()
        


    
    #######################################################################################
    # Initialize class
    #######################################################################################
    
    def init_class_variables(self):
        self.working_path = working_path
        self.series_filename = series_filename # filename (for shot series) to use where index is _0001 but without file extension (.tif)
        
        self.ignore_top_orders = True
        self.autoscale_spectrum = False
        self.shift_wavelengths = True # If self.shift_wavelengths == True then wavelengths are locked to order curves, otherwise to pixels on image
        self.shift_only_bounds_right = True # keeps curves on same location on image during horizontal shift
         
        self.reset_class_variables_soft()
    
    # Gets also called on path change (new folder might contain different calibration data/bounds file). 
    def reset_class_variables_soft(self):
        
        self.first_order_nr = None
        self.total_shift_right = None
        self.total_shift_up = None
        
        self.show_orders = True
        self.autoupdate_spectrum = True
        self.program_mode = None
        self.selected_order_nr = None
        self.selected_order_idx = None
        self.curve_edit_mode = False # TODO: delete?
        
        
        self.photo_array = [] # assumes square image
        self.cumulative_sum = None # for faster vertical integral calculation
        self.bounds_input_array = []
        self.order_plot_points = []
        self.order_poly_coefs = []
        self.order_plot_curves = []
        self.order_bound_points = []
        
        # Spectrum data
        self.orders_x_pixels = []
        self.orders_wavelength_coefficients = []
        self.orders_wavelengths = []
        self.orders_intensities = []
        self.spectrum_curve = None
        
        #self.order_data = {} # necessary because order classes get sorted and indices change
        
        # lists to store calibration data like diffraction order class instances
        self.calib_data_static = None # holds order bounds and shift data
        self.calib_data_dynamic = None # holds points, the order of point classes can change
        
        self.spectrum_total_intensity = 0
    
    #######################################################################################
    # Tkinter window elements
    #######################################################################################
    
    # Create tkinter window elements
    def initialize_window_elements(self):
        
        # Main window
        ######################################
        
        #op_sys_toolbar_height = 0.07 #75 px for 1080p
        #self.root.maxsize(self.root.winfo_screenwidth(), self.root.winfo_screenheight() - round(self.root.winfo_screenheight() * op_sys_toolbar_height))
        
        self.frame_left = tkinter.Frame(self.root)
        self.frame_right = tkinter.Frame(self.root)
        
        self.frame_buttons_input = tkinter.Frame(self.frame_left)
        self.frame_input_wide = tkinter.Frame(self.frame_buttons_input)
        self.frame_input = tkinter.Frame(self.frame_buttons_input)
        self.frame_text = tkinter.Frame(self.frame_buttons_input)
        self.frame_buttons = tkinter.Frame(self.frame_buttons_input)
        
        self.frame_image = tkinter.Frame(self.frame_right)
        self.frame_spectrum = tkinter.Frame(self.frame_right)#, height = self.root.winfo_height() / 3)
        
        
        # Inputs
        ######################################
        
        self.use_sample_var = tkinter.StringVar()
        self.input_path_var = tkinter.StringVar()
        self.input_order_var = tkinter.StringVar()
        #self.integral_width_var = tkinter.StringVar()
        self.shift_orders_up_var = tkinter.StringVar()
        self.shift_orders_right_var = tkinter.StringVar()
        
        input_sample_label = tkinter.Label(self.frame_input_wide, text = 'Use sample:')
        input_sample = tkinter.Entry(self.frame_input_wide, textvariable = self.use_sample_var)
        input_path_label = tkinter.Label(self.frame_input_wide, text = 'Directory path:')
        input_path = tkinter.Entry(self.frame_input_wide, textvariable = self.input_path_var)
        
        input_order_label = tkinter.Label(self.frame_input, text = 'Overwrite first order nr:')
        input_order = tkinter.Entry(self.frame_input, textvariable = self.input_order_var)
        
        input_shift_label_up = tkinter.Label(self.frame_input, text = 'Shift orders up:')
        input_shift_orders_up = tkinter.Entry(self.frame_input, textvariable = self.shift_orders_up_var)
        input_shift_label_right = tkinter.Label(self.frame_input, text = 'Shift orders right:')
        input_shift_orders_right = tkinter.Entry(self.frame_input, textvariable = self.shift_orders_right_var)
        
        btn_save_variables = tkinter.Button(self.frame_input, text = "Save variables", command = self.save_variables)
        btn_reset_shift = tkinter.Button(self.frame_input, text = "Reset shifts", command = self.reset_shift)
        
        
        self.use_sample_var.set(str(self.series_filename))
        self.input_path_var.set(str(self.working_path))
        self.input_order_var.set(str(self.first_order_nr))
        self.shift_orders_up_var.set(0)
        self.shift_orders_right_var.set(0)
        
        
        #input_sample_label.grid(row = 0, column = 0)
        #input_sample.grid(row = 1, column = 0)
        #input_path_label.grid(row = 2, column = 0)
        #input_path.grid(row = 3, column = 0)
        
        input_sample_label.pack(fill='x', expand=1)
        input_sample.pack(fill='x', expand=1)
        input_path_label.pack(fill='x', expand=1)
        input_path.pack(fill='x', expand=1)
        
        input_order_label.grid(row = 4, column = 0)
        input_order.grid(row = 4, column = 1)
        input_shift_label_up.grid(row = 5, column = 0)
        input_shift_orders_up.grid(row = 5, column = 1)
        input_shift_label_right.grid(row = 6, column = 0)
        input_shift_orders_right.grid(row = 6, column = 1)
        btn_save_variables.grid(row = 7, column = 0)
        btn_reset_shift.grid(row = 7, column = 1)
        
        
        # User feedback label
        ######################################
        # Separator and label
        separator = ttk.Separator(self.frame_text, orient='horizontal')
        label_separator_visual = tkinter.Label(self.frame_text, text = 'Program log:')
        separator.pack(fill = 'x')
        label_separator_visual.pack()
        
        # Labels
        self.feedback_text = tkinter.Text(self.frame_text, height = 4, width = 30)
        self.feedback_text.pack(side = 'top')
        
        
        # Important main buttons
        ######################################
        
        # Separator and label
        separator = ttk.Separator(self.frame_buttons, orient='horizontal')
        label_separator_visual = tkinter.Label(self.frame_buttons, text = 'Important main buttons')
        separator.pack(fill = 'x')
        label_separator_visual.pack()
        
        # Create buttons
        btn_autocalibrate = tkinter.Button(self.frame_buttons, text = "Try automatic calibration", command = self.autocalibrate)
        btn_button_save = tkinter.Button(self.frame_buttons, text = "Save calibration data", command = self.save_data)
        btn_ignore_orders = tkinter.Button(self.frame_buttons, text = "Toggle ignore orders near top edge", command = self.ignore_top_orders_fn)
        btn_only_shift_bounds = tkinter.Button(self.frame_buttons, text = "Toggle shift also curve shape", command = self.only_shift_bounds)
        
        btn_autocalibrate.pack()
        btn_button_save.pack()
        btn_ignore_orders.pack()
        btn_only_shift_bounds.pack()
        
       
        
        # Program controlling buttons
        ######################################
        
        # Separator and label
        separator = ttk.Separator(self.frame_buttons, orient='horizontal')
        label_separator_visual = tkinter.Label(self.frame_buttons, text = 'Program control buttons')
        separator.pack(fill = 'x')
        label_separator_visual.pack()
        
        btn_reset_mode = tkinter.Button(self.frame_buttons, text = "Reset program mode", command = self.reset_mode)
        btn_select_order = tkinter.Button(self.frame_buttons, text = "Select/deselect order mode", command = self.select_order)
        btn_orders_mode = tkinter.Button(self.frame_buttons, text = "Diffr order edit mode (armed)", command = self.orders_mode)
        
        btn_add_order = tkinter.Button(self.frame_buttons, text = "Add diffr order", command = self.add_order)
        btn_delete_order = tkinter.Button(self.frame_buttons, text = "Delete diffr order", command = self.delete_order)
        
        
        btn_reset_mode.pack()
        btn_select_order.pack()
        btn_orders_mode.pack()
        
        btn_add_order.pack()
        btn_delete_order.pack()
        
        # Extra features buttons
        ######################################
        
        # Separator and label
        separator = ttk.Separator(self.frame_buttons, orient='horizontal')
        label_separator_visual = tkinter.Label(self.frame_buttons, text = 'Misc. buttons')
        separator.pack(fill = 'x')
        label_separator_visual.pack()
        
        btn_toggle_spectrum_update = tkinter.Button(self.frame_buttons, text = "Toggle spectrum updating (speed)", command = self.toggle_spectrum_update)
        btn_load_order_points = tkinter.Button(self.frame_buttons, text = "Load order points (file)", command = self.load_order_points)
        btn_save_coefs = tkinter.Button(self.frame_buttons, text = "Output stuff (files)", command = self.write_outputs)
        btn_orders_tidy = tkinter.Button(self.frame_buttons, text = "Tidy order points", command = self.orders_tidy)
        btn_shift_wavelengths = tkinter.Button(self.frame_buttons, text = "Toggle shift spectrum wavelengths", command = self.shift_wavelengths_fn)
        
        
        # Show buttons
        btn_toggle_spectrum_update.pack()
        btn_load_order_points.pack()
        btn_save_coefs.pack()
        btn_orders_tidy.pack()
        btn_shift_wavelengths.pack()
        
        
        # Visual options
        ######################################
        
        # Separator and label
        separator = ttk.Separator(self.frame_buttons, orient='horizontal')
        label_separator_visual = tkinter.Label(self.frame_buttons, text = 'Visual options')
        separator.pack(fill = 'x')
        label_separator_visual.pack()
        
        btn_toggle_show_orders = tkinter.Button(self.frame_buttons, text = "Toggle showing orders", command = self.toggle_show_orders)
        btn_spectrum_log = tkinter.Button(self.frame_buttons, text = "Toggle spectrum log scale", command = self.spectrum_toggle_log)
        btn_update_scale = tkinter.Button(self.frame_buttons, text = "Toggle autoupdate spectrum scale", command = self.update_scale_fn)
        
        btn_image_int_down = tkinter.Button(self.frame_buttons, text = "Lower colorbar max x2", command = self.image_int_down)
        btn_image_int_up = tkinter.Button(self.frame_buttons, text = "Raise colorbar max x5", command = self.image_int_up)
        btn_image_int_min_down = tkinter.Button(self.frame_buttons, text = "Lower colorbar min x2", command = self.image_int_min_down)
        btn_image_int_min_up = tkinter.Button(self.frame_buttons, text = "Raise colorbar min x5", command = self.image_int_min_up)
        
        # Show buttons
        btn_toggle_show_orders.pack()
        btn_spectrum_log.pack()
        btn_update_scale.pack()
        
        btn_image_int_down.pack()
        btn_image_int_up.pack()
        btn_image_int_min_down.pack()
        btn_image_int_min_up.pack()
        
        ######################################
        
        
        
        
        
        
        
        self.root.title("Echellogram calibration")
        self.root.minsize(600, 600)
        
        #self.root.update()
        #self.pack_window_elements()
        
    # Draw window elements
    def pack_window_elements(self):
        self.frame_left.pack(fill='both', expand=0, side = 'left')
        self.frame_right.pack(fill='both', expand=1, side = 'left')
        
        self.frame_image.pack(fill='both', expand=1, side = 'top')
        self.frame_spectrum.pack(fill='both', expand=1, side = 'bottom')
        
        
        self.frame_buttons_input.pack(side = 'left')
        self.frame_input_wide.pack(side = 'top', expand = 1, fill = 'x')
        self.frame_input.pack(side = 'top')
        self.frame_text.pack(side = 'top')
        self.frame_buttons.pack(side = 'top')
        
        
    #######################################################################################
    # Load and save calibration data
    #######################################################################################
    
    # Load data from path and initialize program with it
    def load_folder(self, path, do_reset = False, new_sample = None):
        
        # Clear already existing plots
        if do_reset:
            self.clear_plots()
        
        # reset variables
        self.reset_class_variables_soft()
        self.working_path = path
        self.series_filename = new_sample
        
        # Get input data (calibration data, Echellograms etc.)
        try:
            self.load_sample_data()
            
            # Draw matplotlib plot
            self.initialize_plot()
        
        # Catch an exception and print where it came from
        except Exception as e:
            self.series_filename = None
            print(e)
            self.set_feedback(str(e), 10000)
            
            # Print where exactly the exeption was raised
            print("Exception occurred in functiontree:")
            tb = e.__traceback__
            while tb.tb_next:  # Walk to the last traceback frame (where exception was raised)
                print(tb.tb_frame.f_code.co_name)
                tb = tb.tb_next
            print(tb.tb_frame.f_code.co_name)
        
        
    
    def load_sample_data(self):
        
        # average input photos and return the array
        average_array, identificator = prepare_photos(self.series_filename)
        
        # Initialize used sample
        if self.series_filename is None:
            self.series_filename = identificator # use the (random) file series found in the folder
            self.use_sample_var.set(str(self.series_filename))
        
        # sample specified but not found
        if identificator != self.series_filename:
            print('Sample not in input files, found: ' + identificator)
            raise Exception('Sample not in input files, found: ' + identificator)
        
        photo_array, calibration_data, bounds_input_array = process_photo(average_array)
        
        # sort in ascending order nr
        if not bounds_input_array is None:
            bounds_input_array = bounds_input_array[bounds_input_array[:, 0].argsort()]
        
        self.photo_array = photo_array
        self.bounds_input_array = bounds_input_array
        
        # Calculate stuff with given array
        self.max_y_idx = self.photo_array.shape[0] - 1
        self.max_x_idx = self.photo_array.shape[1] - 1
        self.update_cumulsum() # pre-calculate cumulative column sums for integral calculation later
        
        
        # Initialize first order nr and horizontal shift
        if not calibration_data is None:
            if self.first_order_nr is None:
                self.first_order_nr = calibration_data['first_order_nr'] # gets overwritten in self.load_static_data()
            if self.total_shift_right is None:
                self.total_shift_right = calibration_data['total_shift_right']
            if self.total_shift_up is None:
                self.total_shift_up = calibration_data['total_shift_up']
        
        # Load input data
        if self.calib_data_dynamic is None: # only initialize diffr orders once after program start/reset
            self.calib_data_static = []
            self.calib_data_dynamic = []
            self.load_dynamic_data(calibration_data)
            self.load_static_data(calibration_data)
    
    # Calculate cumulative sum for each column separately for faster integral calculation for spectrum
    # Gemini 2.5 Pro invention (and a good one)
    def update_cumulsum(self):
        """Pre-calculates and caches the cumulative sum of photo_array along axis 0."""
        # Pad with a row of zeros at the top for easier cumulsum indexing:
        # sum(S to E inclusive) = cumulsum[E+1] - cumulsum[S]
        cumsum_temp = np.cumsum(self.photo_array, axis=0)
        self.cumulative_sum = np.pad(cumsum_temp, ((1, 0), (0, 0)), 
                                                mode='constant', constant_values=0.)
        
            
    def load_dynamic_data(self, calibration_data):
        
        # initialize calib_data with photo_array bounds for better plotting
        if calibration_data is None:
            self.calib_data_dynamic.append(order_points(image_width = self.photo_array.shape[0]))
            
        # load saved data
        else:
            
            # Iterate over orders
            for order_raw in calibration_data['dynamic']:
                self.calib_data_dynamic.append(order_points(existing_data = order_raw))
            
            # sort orders by average y values
            self.sort_orders(sort_plots = False)
    
    
    # Is meant to be run only once (per folder a.k.a. input files)
    def load_static_data(self, calibration_data):
        
        # If there's data from file then use that, else use previous calibration data
        if not self.bounds_input_array is None:
            
            self.first_order_nr = int(self.bounds_input_array[:, 0].min()) # assumes that input file has correct data #TODO: v3 check
            self.input_order_var.set(str(self.first_order_nr))
            
            self.compile_static_data_from_file()
        
        
        # use previous calibration data
        elif not calibration_data is None: 
            print('_Vertical_points.csv missing. Using previous calibration data')
            self.set_feedback('_Vertical_points.csv missing. Using previous calibration data', 10000)
            
            # Iterate over orders
            self.first_order_nr = math.inf
            for order_raw in calibration_data['static']:
                order_static_class = static_calibration_data(existing_data = order_raw)
                self.calib_data_static.append(order_static_class)
                
                # Register smallest order nr
                if order_static_class.order_nr < self.first_order_nr:
                    self.first_order_nr = order_static_class.order_nr
            
            self.input_order_var.set(str(self.first_order_nr))
            
            if self.first_order_nr == math.inf:
                self.set_feedback('first_order_nr couldn\'t be determined from previous calibration data', 15000)
                raise Exception('first_order_nr couldn\'t be determined from previous calibration data')
            
        else:
            raise Exception('Previous calibration data and vertical points file not found. Can\'t calculate bounds')
    
    # (Re)initializes static calibration data (order nrs and bounds). 
    # Writes into self.calib_data_static only the data that's relevant to drawn orders.
    # When first order nr or number of drawn orders is changed then this is called again.
    def compile_static_data_from_file(self): # TODO: v3 check
        # reset the list, only to contain static data that is relevant for drawn orders
        self.calib_data_static = []
        
        input_array = self.bounds_input_array
        if input_array is None:
            return
        
        dynamic_len = len(self.calib_data_dynamic) # nr of drawn curves
        static_len = input_array.shape[0] # nr of orders in input bounds file
        
        # Let user know that input data wasn't perfect
        if dynamic_len != static_len:
            print(f'Warning! _Vertical_points.csv has {static_len} rows, but there are {dynamic_len} drawn orders.')
            self.set_feedback(f'Warning! _Vertical_points.csv has {static_len} rows, but there are {dynamic_len} drawn orders.', 15000)
        
        # Check if self.first_order_nr is in the file
        idxs = np.where(input_array[:,0] == self.first_order_nr)[0]
        if len(idxs) > 0: # There's match
            idx = idxs[0]
        else:
            print('ERROR! First order nr not in _Vertical_points.csv')
            #self.set_feedback('ERROR! First order nr not in _Vertical_points.csv', 15000)
            raise Exception('ERROR! First order nr not in _Vertical_points.csv')
            
        
        # Iterate over rows in input file array
        for row_idx in range(idx, static_len):
            order_nr = int(input_array[row_idx, 0])
            
            # Ignore the row if the file contains more orders than required
            if order_nr < self.first_order_nr:
                continue
            
            # order isn't created/drawn on the plot
            if len(self.calib_data_static) >= dynamic_len:
                break
            
            
            static_class_instance = static_calibration_data()
            static_class_instance.order_nr = order_nr
            
            px_start = input_array[row_idx, 1]
            px_end = input_array[row_idx, 2]
            static_class_instance.bounds_px = [px_start, px_end]
            static_class_instance.bounds_px_original = [px_start, px_end]
            
            # Initialize self.first_order_nr
            if (self.first_order_nr is None) or (order_nr < self.first_order_nr):
                self.first_order_nr = order_nr
            
            # File contains info about wavelengths
            if input_array.shape[1] > 3:
                wave_start = input_array[row_idx, 3]
                wave_end = input_array[row_idx, 4]
                
                static_class_instance.bounds_wave = [wave_start, wave_end]
                static_class_instance.bounds_wave_original = [wave_start, wave_end]
            
            
            # If you're viewing the code then you can disable the exception throwing, it's here
            # because ordinary user shouldn't use the program without input wavelengths.
            else:
                self.set_feedback('_Vertical_points.csv must contain wavelength columns', 15000)
                raise Exception('_Vertical_points.csv doesn\'t contain wavelength columns')
            
            # File contains also info about middle points
            if input_array.shape[1] > 5:
                px_middle = input_array[row_idx, 5]
                wave_middle = input_array[row_idx, 6]
                static_class_instance.bounds_middle = [px_middle, wave_middle]
                static_class_instance.bounds_middle_orig = [px_middle, wave_middle]
            
            
            self.calib_data_static.append(static_class_instance)
        
        
        # There are too few rows in file, copy last file row for last orders 
        registered_nr = len(self.calib_data_static)
        if dynamic_len > registered_nr:
            print('_Vertical_points.csv has less rows than drawn orders. Last row is copied for bottom drawn orders.')
            self.set_feedback('_Vertical_points.csv has less rows than drawn orders. Last row is copied.', 15000)
            
            last_order = self.calib_data_static[-1]
            
            # Add the remaining orders (copy data from last file row)
            for exess_nr in range(1, dynamic_len - registered_nr + 1):
                order_nr = last_order.order_nr + exess_nr
                
                static_class_instance = static_calibration_data(order_nr = order_nr)
                static_class_instance.bounds_px = last_order.bounds_px
                static_class_instance.bounds_px_original = last_order.bounds_px_original
                static_class_instance.bounds_wave = last_order.bounds_wave
                static_class_instance.bounds_wave_original = last_order.bounds_wave_original
                static_class_instance.bounds_middle = last_order.bounds_middle
                static_class_instance.bounds_middle_original = last_order.bounds_middle_original
                self.calib_data_static.append(static_class_instance)
        
        # In case there has been a shift, recalculate shifted bounds from original bounds
        for order_idx in range(len(self.calib_data_static)):
            self.recalculate_bounds(order_idx)
    
    
    # Save calibration curves and the corresponding points
    def save_data(self):
        
        # decode objects into dictionary in certain key order
        save_dict = {}
        save_dict['first_order_nr'] = self.first_order_nr
        save_dict['total_shift_right'] = self.total_shift_right
        save_dict['total_shift_up'] = self.total_shift_up
        save_dict['static'] = []
        save_dict['dynamic'] = []
        
        
        # Iterate over orders
        for order_idx, order in enumerate(self.calib_data_dynamic):
            
            points_array = order.save_decode()
            save_dict['dynamic'].append(points_array)
        
        for order in self.calib_data_static:
            static_dict = order.save_decode()
            save_dict['static'].append(static_dict)
        
        
        # Save as JSON readable output
        with open(output_path + '_Calibration_data.json', 'w') as file: # '_' in front to find easily among Echellograms
            json.dump(save_dict, file, sort_keys = False, indent = 2)
        
        self.set_feedback('Calibration data saved')
    
    
    #######################################################################################
    # Button functions
    #######################################################################################
    
    def save_variables(self):
        
        # E.g. reset selected order
        self.reset_mode(button_call = False)
        
        new_sample = self.use_sample_var.get()
        path = self.input_path_var.get()
        first_order_nr = int(self.input_order_var.get())
        #self.integral_width = float(self.integral_width_var.get())
        shift_orders_amount_up = float(self.shift_orders_up_var.get())
        shift_orders_amount_right = float(self.shift_orders_right_var.get())
        
        # Empty the variables
        #self.input_path_var.set('')
        #self.input_order_var.set('')
        #self.integral_width_var.set('')
        self.shift_orders_up_var.set(0)
        self.shift_orders_right_var.set(0)
        
        # New folder, do almost everything from scratch, ignore other saved variables
        if path != self.working_path:
            self.series_filename = new_sample
            self.update_path(path, new_sample = new_sample)
            return
        
        # Use new sample, load data again and draw plots again
        if new_sample != self.series_filename:
            self.update_sample(new_sample)
            return
        
        
        # First order nr changes, so re-compile self.calib_data_static
        if (self.first_order_nr != first_order_nr):
            
            # Check that the entered value isn't too small
            file_first_order_nr = int(self.bounds_input_array[:, 0].min()) # TODO: v3 check
            if first_order_nr >= file_first_order_nr:
                self.first_order_nr = first_order_nr
                self.compile_static_data_from_file()
                
            else:
                self.input_order_var.set(str(self.first_order_nr))
                print('Entered first order nr not in _Vertical_points.csv')
                self.set_feedback('Entered first order nr not in _Vertical_points.csv', 10000)
            
            
        
        
        # Shift order curves and bounds. If shift is 0 then returns immediately.
        self.shift_orders_up(shift_orders_amount_up)
        self.shift_orders_right(shift_orders_amount_right)
        
        
        self.update_all()
        self.set_feedback('Input variables saved')
        
    
    def update_path(self, path, new_sample = None):
        self.set_feedback('New folder loaded')
        
        # Check if path has slash at the end
        if path[-1] != system_path_symbol:
            path += system_path_symbol
            self.input_path_var.set(path)
        
        # Update paths
        global input_photos_path, input_data_path, averaged_path, output_path, spectrum_path
        if script_manual_mode: # Paths in custom locations
            input_photos_path = path + 'Input_photos' + system_path_symbol
            input_data_path = path + 'Input_data' + system_path_symbol
            averaged_path = path + 'Averaged' + system_path_symbol
            output_path = path + 'Output' + system_path_symbol
            spectrum_path = path + 'Spectra' + system_path_symbol
        else: # Paths are the same (input = output)
            input_photos_path = path
            input_data_path = path
            averaged_path = path
            output_path = path
            spectrum_path = path
        
        self.load_folder(path, do_reset = True, new_sample = new_sample)
    
    def update_sample(self, new_sample):
        # Load new sample data
        old_sample = self.series_filename
        self.series_filename = new_sample
        try:
            self.load_sample_data()
        except Exception as e:
            self.series_filename = old_sample
            print(e)
            print('Keeping old sample')
            self.set_feedback(str(e), 10000)
        
        # Draw plots again
        self.initialize_plot(do_reset = True)
    
    
    # Shift orders horizontally and vertically until spectrum has max value and 
    # spectral lines are at correct wavelengths
    def autocalibrate(self):
        best_shift, best_int = self.autocalibrate_vertical()
        self.set_feedback('Optimal vertical shift found')
        self.autocalibrate_horizontal()
    
    
    # Shift orders vertically until spectrum has max value
    # TODO: sometimes stops sub-optimally (local minimum or flawed logic?)
    def autocalibrate_vertical(self):
        
        best_int = self.spectrum_total_intensity
        
        # Try shifting curves 1 px up and 1 px down, then 2 px up (from start) then down etc
        int_dict = {1 : 0, -1 : 0}
        shift_orders_amount = 0.75
        step_size = shift_orders_amount
        direction = 1 # 1 is up on image (lower y-value)
        best_shift = 0
        total_shift = 0
        while shift_orders_amount < 20:
            
            # There was no better shift in either direction for 2 px
            if (int_dict[1] >= 2) and (int_dict[-1] >= 2):
                break
            
            # This direction isn't good, there have been 2 subsequent worse shifts
            if int_dict[direction] >= 2:
                
                # Keep shifting in good direction ca one pixel at a time
                shift_orders_amount = 0.75
                step_size = 0
                
                # Ignore current wrong direction
                direction *= -1
                continue
            
            shift_amount = shift_orders_amount * direction
            new_int = self.autocalibrate_shift_vertical(shift_amount)
            
            total_shift += shift_amount
            
            # Save better shift or worse shift
            if new_int > best_int:
                best_shift = total_shift
                best_int = new_int
            else:
                int_dict[direction] += 1
            
            # Try other direction
            direction *= -1
            
            # Both directions done, try bigger shift
            if direction == 1:
                shift_orders_amount += step_size
        
        # Shift back to optimal location
        if total_shift != best_shift:
            shift_amount = best_shift - total_shift
            self.autocalibrate_shift_vertical(shift_amount)
        
        return best_shift, best_int
    
    
    def autocalibrate_shift_vertical(self, shift_amount):
        self.shift_orders_up(shift_amount)
        
        #self.update_all() # too slow
        self.update_point_instances()
        self.update_order_curves()
        self.calculate_all_bounds(initialize_x = True)
        self.update_spectrum(autoscale = True) # calculated spectrum int
        
        new_int = self.spectrum_total_intensity
        
        return new_int
    
    def autocalibrate_horizontal(self):
        pass # TODO
    
    def autocalibrate_shift_horizontal(self, shift_amount):
        self.shift_orders_right(shift_amount)
        
        self.update_point_instances()
        self.update_order_curves()
        self.calculate_all_bounds(initialize_x = True)
        self.update_spectrum(autoscale = True)
    
    
    
    # If true then shifting orders horizontally also shifts wavelengths
    # If self.shift_wavelengths == True then wavelengths are locked to order curves, otherwise to pixels on image
    def shift_wavelengths_fn(self):
        self.shift_wavelengths = not self.shift_wavelengths
        # TODO: update spectrum
        self.set_feedback('Shift spectrum wavelengths (wavelengths locked to orders): ' + str(self.shift_wavelengths))
    
    # Enable/disable orders that are too close to image top edge
    def ignore_top_orders_fn(self):
        self.ignore_top_orders = not self.ignore_top_orders
        self.check_top_orders_proximity()
        self.set_feedback('Ignore out of bounds orders: ' + str(self.ignore_top_orders))
    
    
    # Toggles whether horizontal shift moves only bound points or also order curve points
    def only_shift_bounds(self):
        self.shift_only_bounds_right = not self.shift_only_bounds_right
        self.set_feedback('Only shift bounds right (vs curves): ' + str(self.shift_only_bounds_right))
    
    
    # Reset program mode
    def reset_mode(self, button_call = True):
        
        # Reset selected order color
        self.update_one_order_curve_color(self.selected_order_idx, set_color = 'r')
        
        self.program_mode = None
        self.selected_order_idx = None
        self.selected_order_nr = None
        self.curve_edit_mode = False
        
        
        if button_call:
            self.set_feedback('Mode: reset')
    
    # In this mode you can (de)select a diffraction order to be modified in orders mode
    def select_order(self):
        self.program_mode = 'select'
        self.curve_edit_mode = False
    
    
        self.set_feedback('Mode: select')
        
    
    # In this mode you can move the selected diffraction order points, no effect if no order selected
    def orders_mode(self):
        
        if self.selected_order_nr is None:
            self.set_feedback('Error: No order selected', 8000)
            return
        
        self.curve_edit_mode = True
        self.program_mode = 'orders'
        
        self.show_orders = True
        #self.update_order_curves()
        self.hide_unhide_orders()
        
        self.set_feedback('Mode: horizontal curves')
    
    
    # Read in csv file and overwrite order points based on these
    def load_order_points(self):
        
        all_points = np.loadtxt(input_data_path + '_Order_points.csv', delimiter=',', skiprows = 1)
        
        for order_idx in range(len(self.calib_data_dynamic)):
            order = self.calib_data_dynamic[order_idx]
            
            # get order_nr row index from data
            order_nr = self.calib_data_static[order_idx].order_nr
            row_idxs = np.where(all_points[:,0] == order_nr)[0] # get indices of rows where order nr is the same
            
            # Get bounds
            if len(row_idxs) > 0: # There's match
                row_idx = row_idxs[0]
                order.points[0].x = all_points[row_idx, 1]
                order.points[0].y = all_points[row_idx, 2]
                order.points[1].x = all_points[row_idx, 3]
                order.points[1].y = all_points[row_idx, 4]
                order.points[2].x = all_points[row_idx, 5]
                order.points[2].y = all_points[row_idx, 6]
        
        self.update_all()
        self.set_feedback('Order points loaded')
        
    def write_outputs(self):
       self.output_coefs()
       self.output_order_points()
       self.output_spectrum()
       self.set_feedback('Output written')
   
    # Save the polynomial coefficients of diffraction orders into output
    def output_coefs(self):
        
        with open(output_path + '_Order_coefficients.csv', 'w') as file:
            file.write('Order_nr,Coef1,Coef2,Coef3...\n') # headers
            
            for order_idx in range(len(self.calib_data_dynamic)):
                order_nr = self.calib_data_static[order_idx].order_nr
                coefs = self.order_poly_coefs[order_idx]
                file.write(str(order_nr))
                
                for coef in coefs:
                    file.write(',' + str(coef))
                
                file.write('\n')
    
    
    # Write the three points x and y coordinates for each order to extra file
    def output_order_points(self):
        with open(output_path + '_Order_points.csv', 'w') as file:
            file.write('Order_nr,Point0_x,Point0_y,Point1_x,Point1_y,Point2_x,Point2_y\n') # headers
            
            for idx in range(len(self.calib_data_dynamic)):
                order_nr = self.calib_data_static[idx].order_nr
                file.write(str(order_nr))
                
                points = self.calib_data_dynamic[idx].points
                for point in points:
                    file.write(',' + str(point.x))
                    file.write(',' + str(point.y))
                
                file.write('\n')
    
    # Write the spectrum x and y data to extra file
    def output_spectrum(self):
        #x_data = self.spectrum_ax.get_xdata()
        #y_data = self.spectrum_ax.get_ydata()
        x_values, z_values = self.compile_full_spectrum()
        order_nrs = self.get_spectrum_order_nrs()
        data = np.column_stack((x_values, z_values, order_nrs))
        np.savetxt(output_path + '_Spectrum.csv', data, fmt = '%.8e', delimiter = ',', comments = '', header = 'Wavelength (nm),Intensity,Order nr')
        
    
    
    # Overwrite the order points with the ones calculated from polynomials
    def orders_tidy(self):
        start_x = 35
        center_x = math.floor(self.photo_array.shape[0] / 2)
        end_x = self.photo_array.shape[0] - 35
        
        for idx in range(len(self.calib_data_dynamic)):
            order = self.calib_data_dynamic[idx]
            
            order.points[0].x = start_x
            order.points[0].y = poly_func_value(start_x, self.order_poly_coefs[idx])
            order.points[1].x = center_x
            order.points[1].y = poly_func_value(center_x, self.order_poly_coefs[idx])
            order.points[2].x = end_x
            order.points[2].y = poly_func_value(end_x, self.order_poly_coefs[idx])
            
    
        self.update_all()
    
    def add_order(self):
        
        # Show orders
        self.show_orders = True
        self.hide_unhide_orders()
        
        diffr_order = order_points(self.photo_array.shape[0])
        self.calib_data_dynamic.append(diffr_order)
        
        # Recompile self.calib_data_static with relevant data
        self.compile_static_data_from_file()
        # Calculate the bounds for that order
        #self.initialize_bounds(self.selected_order_idx)
        
        self.plot_order(diffr_order, color = 'white')
        
        # sort orders by average y values
        #self.calib_data_dynamic.sort(key=lambda x: x.avg_y, reverse=True)
        
        self.selected_order_nr = diffr_order
        self.selected_order_idx = len(self.calib_data_dynamic) - 1
        
        
        self.draw_bounds(self.selected_order_idx)
        
        self.sort_orders()
        
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
        
        #self.order_bound_points[idx]
        
        #self.update_order_curves()
        self.update_spectrum(autoscale = True)
        
        
        # Draw plot again and wait for drawing to finish
        self.canvas_spectrum.draw()
        self.canvas_spectrum.flush_events()
        
        self.set_feedback('Order added and selected', 1000)
    
    def delete_order(self):
        
        if self.selected_order_idx is None:
            self.set_feedback('No order selected', 8000)
            return
        
        # Show orders
        self.show_orders = True
        self.hide_unhide_orders()
        
        selected_idx = self.selected_order_idx
        order_nr = self.calib_data_static[selected_idx].order_nr
        
        del self.calib_data_dynamic[selected_idx]
        
        # Recompile self.calib_data_static with relevant data
        self.compile_static_data_from_file()
        
        # Deselect the order
        self.selected_order_nr = None
        self.selected_order_idx = None
        
        # Draw plots again
        self.initialize_plot(do_reset = True)
        self.update_all()
        
        #self.update_spectrum(autoscale = True)
        
        self.reset_mode(button_call = False)
        
        self.set_feedback('Order deleted: ' + str(order_nr), 5000)
    
    def spectrum_toggle_log(self):
        current_scale = self.spectrum_ax.get_yscale()
        if current_scale == 'symlog':
            self.spectrum_ax.set_yscale('linear')
            current_scale = 'linear'
        else:
            self.spectrum_ax.set_yscale('symlog') # symlog for 0-values
            current_scale = 'symlog'
        
        # Draw plot again and wait for drawing to finish
        self.canvas_spectrum.draw()
        self.canvas_spectrum.flush_events()
        
        self.set_feedback('New spectrum scale: ' + current_scale)
    
    def toggle_spectrum_update(self):
        self.autoupdate_spectrum = not self.autoupdate_spectrum
        
        if self.autoupdate_spectrum:
            # Show orders and update the data
            self.show_orders = True
            self.update_order_curves()
            self.hide_unhide_orders()
        
        self.set_feedback('Spectrum updating: ' + str(self.autoupdate_spectrum))
    
    def update_scale_fn(self):
        self.autoscale_spectrum = not self.autoscale_spectrum
        self.set_feedback('Spectrum autoscale: ' + str(self.autoscale_spectrum))
    
    
    def toggle_show_orders(self):
        self.show_orders = not self.show_orders
        self.hide_unhide_orders()
        self.set_feedback('Show orders: ' + str(self.show_orders))
    
    def image_int_down(self):
        old_min = self.cbar.vmin
        old_max = self.cbar.vmax
        self.image.set_clim(old_min, old_max / 2)
        
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
        
        self.set_feedback('New max: ' + '{:.3e}'.format(old_max / 2), 5000)
    
    def image_int_up(self):
        old_min = self.cbar.vmin
        old_max = self.cbar.vmax
        self.image.set_clim(old_min, old_max * 5)
        
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
        
        self.set_feedback('New max: ' + '{:.3e}'.format(old_max * 5), 5000)
    
    def image_int_min_down(self):
        old_min = self.cbar.vmin
        old_max = self.cbar.vmax
        self.image.set_clim(old_min / 2, old_max)
        
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
        
        self.set_feedback('New min: ' + '{:.3e}'.format(old_min / 2), 5000)
    
    def image_int_min_up(self):
        old_min = self.cbar.vmin
        old_max = self.cbar.vmax
        self.image.set_clim(old_min * 5, old_max)
        
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
        
        self.set_feedback('New min: ' + '{:.3e}'.format(old_min * 5), 5000)
    
    #######################################################################################
    # Callbacks
    #######################################################################################
    
    # Do stuff when tkinter window is resized
    
    def resize_callback(self, event):
        
        new_plot_width = event.width / self.photo_fig.dpi
        new_plot_height = event.height / self.photo_fig.dpi
        self.photo_fig.set_size_inches(new_plot_width, new_plot_height, forward = True)
        
        self.photo_fig.subplots_adjust(left = 0.1, right = 0.9, top = 0.9, bottom = 0.1, wspace = 0.3)
        
        pos = self.plot_ax.get_position()
        self.plot_ax.set_position([pos.x0, pos.y0, pos.width * 1.5, pos.height])
        
        self.canvas.draw_idle()
        
    
    def plot_click_callback(self, event):
        #print(event.xdata, event.ydata)
        
        try:
            click_point = point_class(event.xdata, y = event.ydata)
        except ValueError: # Clicked outside of the plot
            return
        
        # Select the diffraction order which's point is closest to the click
        if self.program_mode == 'select':
            self.select_click(click_point)
            
            
        # Modify horizontal curves
        elif self.program_mode == 'orders':
            self.order_click(click_point)
            
        
    # Select order
    def select_click(self, click_point):
        if not self.show_orders:
            return
        
        # Iterate over diffraction orders, select the point closest to the click
        min_distance = math.inf
        for idx in range(len(self.calib_data_dynamic)):
            order = self.calib_data_dynamic[idx]
            
            # iterate over points in order, save best point
            for idx2 in range(len(order.points)):
                distance = order.points[idx2].distance(click_point)
                if distance < min_distance:
                    best_order_idx = idx
                    min_distance = distance
        
        # Deselect
        if self.selected_order_idx == best_order_idx:
            
            self.selected_order_idx = None
            self.selected_order_nr = None
            
            self.update_one_order_curve_color(best_order_idx, set_color = 'r')
            self.set_feedback('Order ' + str(best_order_idx + self.first_order_nr) + ' deselected')
        
        # Select
        else:
            # paint last selection red
            self.update_one_order_curve_color(self.selected_order_idx, set_color = 'r')
            
            self.selected_order_idx = best_order_idx
            self.selected_order_nr = self.calib_data_static[best_order_idx].order_nr
            
            self.update_one_order_curve_color(best_order_idx, set_color = 'white')
            self.set_feedback('Order ' + str(best_order_idx + self.first_order_nr) + ' selected')
    
    
    def order_click(self, click_point):
        
        if self.selected_order_nr is None:
            self.set_feedback('Order mode but none selected')
            return
        
        if not self.show_orders:
            return
        
        order = self.calib_data_dynamic[self.selected_order_idx]
        
        # Iterate over points of the selected orded
        min_distance = math.inf
        for idx in range(len(order.points)):
            distance = order.points[idx].distance(click_point)
            if distance < min_distance:
                best_point_idx = idx
                min_distance = distance
                best_point = order.points[idx]
        
        #print('best: ', best_point.group, best_point_idx, best_point.x, best_point.y)
        
        # edit point
        click_point.group = best_point.group
        self.calib_data_dynamic[self.selected_order_idx].points[best_point_idx] = click_point # TODO: delete hack 
        self.calib_data_dynamic[self.selected_order_idx].update()
        
        
        # sort orders by average y values, this might change order nrs
        self.sort_orders()
        
        # Calculate the bounds for that order
        self.recalculate_bounds(self.selected_order_idx)
        
        # TODO: update only the selected order and all bounds
        self.update_order_curves()
        
        # Update spectrum data with reordered orders
        self.update_spectrum_data()
        
        # Redraw plot
        if self.autoupdate_spectrum and (not self.program_mode is None):
            self.update_spectrum()
       
    
    def shift_orders_up(self, shift_amount = 0, button_call = True):
        if shift_amount == 0:
            return
        
        # Register total shift
        self.total_shift_up += shift_amount
        
        for order_idx in range(len(self.calib_data_dynamic)):
            order = self.calib_data_dynamic[order_idx]
            
            for point in order.points:
                point.y -= shift_amount # shift up, so coordinate decreases
        
        self.set_feedback('Orders shifted up by ' + str(shift_amount) + ' px')
        
        # update visuals
        if button_call:
            self.update_point_instances()
            self.update_order_curves()
            self.calculate_all_bounds(initialize_x = True)
            
            # Recalculate intensities
            for order_idx in range(len(self.calib_data_dynamic)):
                self.update_spectrum_data_list(order_idx, self.orders_intensities, self.calc_order_intensities)
            
            self.update_spectrum(autoscale = True)
    
    def shift_orders_right(self, shift_amount = 0, button_call = True):
        if shift_amount == 0:
            return
        
        # Register total shift
        self.total_shift_right += shift_amount
        
        for order_idx in range(len(self.calib_data_dynamic)):
            
            if not self.shift_only_bounds_right:
                # shift order points
                order = self.calib_data_dynamic[order_idx]
                for point in order.points:
                    point.x += shift_amount # shift right
            
            self.recalculate_bounds(order_idx, last_shift = shift_amount)
            
        
        self.set_feedback('Orders shifted right by ' + str(shift_amount) + ' px')
        
        # update visuals
        if button_call:
            self.update_point_instances()
            self.update_order_curves()
            self.calculate_all_bounds(initialize_x = True)
            self.update_spectrum_data()
            self.update_spectrum(autoscale = True)
    
    
    def recalculate_bounds(self, order_idx, last_shift = 0): # TODO for v3: check middle px and wl calculations
        
        # Short-circuit the function if there's no need to properly calculate stuff again
        if (self.total_shift_right == 0) and (last_shift == 0):
            return self.calib_data_static[order_idx].bounds_px
        
        
        # shift order bounds (pixels)
        [orig_px_left, orig_px_right] = self.calib_data_static[order_idx].bounds_px_original
        left_px = clip(orig_px_left + self.total_shift_right, 0, self.max_x_idx)
        right_px = clip(orig_px_right + self.total_shift_right, 0, self.max_x_idx)
        
        [orig_px_middle, orig_wave_middle] = self.calib_data_static[order_idx].bounds_middle_orig
        [px_middle, wave_middle] = self.calib_data_static[order_idx].bounds_middle
        
        if not px_middle is None:
            px_middle = clip(orig_px_middle + self.total_shift_right, 0, self.max_x_idx)
        
        # Shift wavelengths. Under normal conditions this isn't done because wavelenghts should be locked to the bounds.
        # If self.shift_wavelengths == True then wavelengths are locked to order curves, otherwise to pixels on image
        if not self.shift_wavelengths:
            [orig_wave_left, orig_wave_right] = self.calib_data_static[order_idx].bounds_wave_original
            
            # TODO: wavelength changes according to quadratic fn
            left_wave = linear_regression(left_px, orig_px_left, orig_px_right, orig_wave_left, orig_wave_right)
            right_wave = linear_regression(right_px, orig_px_left, orig_px_right, orig_wave_left, orig_wave_right)
            self.calib_data_static[order_idx].bounds_wave = [left_wave, right_wave]
            
            if not wave_middle is None:
                wave_middle = linear_regression(px_middle, orig_px_left, orig_px_middle, orig_wave_left, orig_wave_middle)
        
        # overwrite previous values
        self.calib_data_static[order_idx].bounds_px = [left_px, right_px]
        self.calib_data_static[order_idx].bounds_middle = [px_middle, wave_middle]
        
        return [left_px, right_px]
    
    
    # Reads total shift up and right and resets the orders location, so that total shift is 0
    def reset_shift(self):
        up = self.total_shift_up
        right = self.total_shift_right
        
        self.shift_orders_up(shift_amount = -self.total_shift_up, button_call = False)
        self.shift_orders_right(shift_amount = -self.total_shift_right, button_call = False)
        
        # Redraw stuff
        self.update_point_instances()
        self.update_order_curves()
        self.calculate_all_bounds(initialize_x = True)
        self.update_spectrum_data()
        self.update_spectrum(autoscale = True)
        
        self.set_feedback('Shifts reset. Relative shifts (up, right): ' + str(up) + ', ' + str(right))
        
        
    
    #######################################################################################
    # Initialize Matplotlib plot
    #######################################################################################
    
    
    def initialize_plot(self, do_reset = False):
        
        if do_reset:
            self.clear_plots()
        
        # Draw main image
        self.draw_plot(identificator = '')
        
        # Draw calibration curves
        self.draw_calibr_curves()
        
        # Initialize bounds of diffraction orders, requires plots to be drawn
        self.calculate_all_bounds(initialize_x = True)
        
        # TODO: fix bounds not being drawn initially (because self.calculate_bounds assumes plotted curves)
        self.update_order_curves()
        
        # Disable orders that are too close to the top edge
        self.check_top_orders_proximity()
        
        # Calculate x-points, wavelengths, quadratic coefficients for the conversion, corresponding intensities
        self.update_spectrum_data()
        
        # Compile spectrum from diffraction orders and the bounds
        self.draw_spectrum()
        
        # Draw plots again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events()
        self.canvas_spectrum.draw()
        self.canvas_spectrum.flush_events() 
    
    
    
    def clear_plots(self):
        # Clear plot
        #self.photo_fig.clf()
        self.canvas._tkcanvas.destroy()
        self.plot_toolbar.destroy()
        
        # Clear spectrum
        self.canvas_spectrum._tkcanvas.destroy()
        self.spectrum_toolbar.destroy()
        #self.canvas_spectrum
        #self.spectrum_fig
        #self.spectrum_ax
        #self.spectrum_curve
        
        # Clear saved line objects
        self.order_plot_curves = []
        self.order_plot_points = []
        self.order_poly_coefs = []
        self.order_bound_points = []
    
    # Draw main heatmap
    
    def draw_plot(self, identificator = ''):
        self.photo_fig = plt.figure()
        
        # Attach figure to the Tkinter frame
        self.canvas = FigureCanvasTkAgg(self.photo_fig, self.frame_image)
        self.plot_toolbar = NavigationToolbar2Tk(self.canvas, self.frame_image)
        self.plot_toolbar.update()
        self.canvas._tkcanvas.pack(fill='both', expand=1, side = 'top')
        
        # x-axis to the top of the image
        self.plot_ax = self.photo_fig.gca()
        self.plot_ax.xaxis.tick_top()
        
        
        # Show image and colorbar
        self.image = plt.imshow(self.photo_array, norm = 'log', aspect = 'auto') # symlog scale min can't be modified
        self.cbar = plt.colorbar()
        self.photo_fig.tight_layout()
        
        self.canvas.mpl_connect('button_press_event', self.plot_click_callback)
        #matplotlib.backend_bases.MouseEvent
        
        
        
    
    def draw_calibr_curves(self):
        self.plot_orders()
        self.draw_all_bounds()
    
    
    # Iterate over diffraction orders and plot them
    
    def plot_orders(self):
        x_values = np.arange(self.photo_array.shape[0])
        for idx in range(len(self.calib_data_dynamic)):
            order = self.calib_data_dynamic[idx]
            self.plot_order(order, x_values)
        
    
    def plot_order(self, order, x_values = None, color = 'r'):
        if x_values is None:
            x_values = np.arange(self.photo_array.shape[0])
        
        
        # Get line data
        curve_array, poly_coefs = get_polynomial_points(order, self.photo_array.shape[0])
        self.order_poly_coefs.append(poly_coefs)
        
        # Plot curve
        curve, = self.plot_ax.plot(x_values, curve_array, color)
        self.order_plot_curves.append(curve)
        
        # Plot calibration points
        curve_points, = self.plot_ax.plot(order.xlist, order.ylist, color = mcolors.CSS4_COLORS['peru'], marker = 'o', linestyle = '', markersize = 4)
        self.order_plot_points.append(curve_points)
        
    
    # Draw points for the bounds even if they aren't calculated correctly yet
    
    def draw_all_bounds(self):
        for order_idx in range(len(self.calib_data_dynamic)):
            self.draw_bounds(order_idx)
    
    def draw_bounds(self, order_idx):
        x_start, x_end, y_start, y_end = self.calculate_bounds(order_idx, initialize_x = True)
        
        bound_points, = self.plot_ax.plot([x_start, x_end], [y_start, y_end], color = 'k', marker = 'x', linestyle = '', markersize = 2)
        self.order_bound_points.append(bound_points)
    
    
    def draw_spectrum(self):
        self.spectrum_fig = plt.figure()
        
        # Attach figure to the Tkinter frame
        self.canvas_spectrum = FigureCanvasTkAgg(self.spectrum_fig, self.frame_spectrum)
        self.spectrum_toolbar = NavigationToolbar2Tk(self.canvas_spectrum, self.frame_spectrum)
        self.spectrum_toolbar.update()
        self.canvas_spectrum._tkcanvas.pack(fill='both', expand=1, side = 'top')
        
        self.spectrum_ax = self.spectrum_fig.gca()
        
        # Plot curve
        x_values, z_values = self.compile_full_spectrum()
        self.spectrum_curve, = self.spectrum_ax.plot(x_values, z_values, 'tab:pink')
        
        # rescale to fit
        self.spectrum_ax.set_xlim(min(x_values), max(x_values))
        self.spectrum_ax.set_ylim(min(z_values), max(z_values))
        
    
    #######################################################################################
    # Update visual things
    #######################################################################################
    
    def clear_feedback(self):
        #self.feedback_text.config(text = '')
        self.feedback_text.delete("1.0", tkinter.END)
    
    def set_feedback(self, string, delay = 5000):
        delay = clip(delay, 2000, 10000)
        
        self.feedback_text.insert("1.0", string + '\n')
        
        #self.feedback_text.config(text = string)
        self.feedback_text.after(delay, self.clear_feedback) # clear feedback after delay
    
    
    
    def update_all(self):
        
        # Deselect order
        self.selected_order_idx = None
        
        self.update_point_instances()
        
        # sort orders by average y values
        self.sort_orders()
        
        self.update_order_curves()
        
        
        self.calculate_all_bounds(initialize_x = True)
        
        # Disable orders that are too close to the top edge
        self.check_top_orders_proximity()
        
        self.update_spectrum_data()
        self.update_spectrum(autoscale = True)
        
        # Draw things again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events()
        self.canvas_spectrum.draw()
        self.canvas_spectrum.flush_events()
    
    # Update instances of point classes
    
    def update_point_instances(self):
        for idx in range(len(self.calib_data_dynamic)):
            self.calib_data_dynamic[idx].update()
    
    
    
    # Redraw horizontal calibration curves
    def update_order_curves(self):
        
        if not self.show_orders:
            return
        
        # Iterate over diffraction orders
        for order_idx in range(len(self.calib_data_dynamic)):
            self.update_one_order_curve(order_idx)
            
        self.update_all_bounds()
            
            
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
    
    def update_one_order_curve_color(self, order_idx, single_call = True, set_color = 'r'):
        if order_idx is None:
            return
        
        if self.order_plot_curves[order_idx].get_color() == set_color:
            return
        
        # Change color and redraw
        self.order_plot_curves[order_idx].set_color(set_color)
        self.canvas.draw()
        self.canvas.flush_events() 
        
    
    def update_one_order_curve(self, order_idx, single_call = False, recalculate = True, set_color = None):
        
        if order_idx is None:
            return
        
        if recalculate:
            order = self.calib_data_dynamic[order_idx]
            
            # curve
            curve_array, poly_coefs = get_polynomial_points(order, self.photo_array.shape[0])
            self.order_poly_coefs[order_idx] = poly_coefs
            self.order_plot_curves[order_idx].set_ydata(curve_array)
            
            # Calculate the new bounds for that order
            self.recalculate_bounds(order_idx)
            #self.initialize_bounds(order_idx)
            
            # points
            self.order_plot_points[order_idx].set_xdata(order.xlist)
            self.order_plot_points[order_idx].set_ydata(order.ylist)
        
        if not set_color is None:
            self.order_plot_curves[order_idx].set_color(set_color)
        
        # Draw plot again and wait for drawing to finish
        if single_call:
            self.canvas.draw()
            self.canvas.flush_events() 
        
    
    def update_all_bounds(self):
        for idx in range(len(self.calib_data_dynamic)):
            self.update_one_bounds(idx)
            
    # Re-draw bounds
    def update_one_bounds(self, order_idx):
        x_start, x_end, y_start, y_end = self.calculate_bounds(order_idx, initialize_x = True)
        self.order_bound_points[order_idx].set_xdata([x_start, x_end])
        self.order_bound_points[order_idx].set_ydata([y_start, y_end])
        
    
    def update_spectrum(self, autoscale = False, force_scale = False):
        if self.spectrum_curve is None:
            return
        
        x_values, z_values = self.compile_full_spectrum()
        
        # Plot curve
        self.spectrum_curve.set_xdata(x_values)
        self.spectrum_curve.set_ydata(z_values)
        
        # rescale to fit
        if force_scale or (self.autoscale_spectrum and autoscale):
            self.spectrum_ax.set_xlim(min(x_values), max(x_values))
            self.spectrum_ax.set_ylim(min(z_values), max(z_values))
        
        # Draw plot again and wait for drawing to finish
        self.canvas_spectrum.draw()
        self.canvas_spectrum.flush_events() 
    
    def hide_unhide_orders(self):
        
        # Iterate over diffraction orders
        for idx in range(len(self.calib_data_dynamic)):
            #order = self.calib_data_dynamic[idx]
            
            self.order_plot_curves[idx].set_visible(self.show_orders)
            self.order_plot_points[idx].set_visible(self.show_orders)
            self.order_bound_points[idx].set_visible(self.show_orders)
        
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
            
    
    
    #######################################################################################
    # Misc
    #######################################################################################
    
    
    def calculate_all_bounds(self, initialize_x = False):
        for idx in range(len(self.calib_data_dynamic)):
            self.calculate_bounds(idx, initialize_x = initialize_x)
    
    def calculate_bounds(self, order_idx, initialize_x = False):
        order = self.calib_data_dynamic[order_idx]
        [px_start, px_end] = self.calib_data_static[order_idx].bounds_px
        
        if initialize_x or (px_start is None) or (px_end is None):
            [px_start, px_end] = self.recalculate_bounds(order_idx) #self.initialize_bounds(order_idx)
        
        coefs = np.polynomial.polynomial.polyfit(order.xlist, order.ylist, 2)
        y_start = poly_func_value(px_start, coefs)
        y_end = poly_func_value(px_end, coefs)
        
        return px_start, px_end, y_start, y_end
    
    '''
    # Calculate bounds by the crossing of curves and save into order points class
    def initialize_bounds(self, order_idx):
        nr_pixels = self.photo_array.shape[0]
        
        curve_x = self.order_plot_curves[order_idx].get_xdata()
        curve_y = self.order_plot_curves[order_idx].get_ydata()
        
        # sort the arrays in increasing x value
        curve_x, curve_y = sort_related_arrays(curve_x, curve_y) 
        
        
        
        # Initialize bounds only once
        if self.calib_data_static[order_idx].bounds_px_original[0] is None: 
            
            # TODO
            #self.order_data[order_idx]
            
            # get order_nr row index from data
            order_nr = self.calib_data_static[order_idx].order_nr
            row_idxs = np.where(self.bounds_input_array[:,0] == order_nr)[0] # get indices of rows where order nr is the same
            
            # Get bounds
            if len(row_idxs) > 0: # There's match
                row_idx = row_idxs[0]
                px_start = self.bounds_input_array[row_idx, 1]
                px_end = self.bounds_input_array[row_idx, 2]
                
                # File contains info about wavelengths
                if self.bounds_input_array.shape[1] > 3:
                    wave_start = self.bounds_input_array[row_idx, 3]
                    wave_end = self.bounds_input_array[row_idx, 4]
                    self.calib_data_static[order_idx].bounds_wave = [wave_start, wave_end]
                    self.calib_data_static[order_idx].bounds_wave_original = [wave_start, wave_end]
            
            else: # no match, use photo bounds
                px_start = 0
                px_end = nr_pixels
            
            self.calib_data_static[order_idx].bounds_px = [px_start, px_end]
            self.calib_data_static[order_idx].bounds_px_original = [px_start, px_end]
        
        
        
        # In case of shift, calculate new bounds and wavelengths
        [px_start, px_end] = self.recalculate_bounds(order_idx)
        
        return px_start, px_end
    '''
    
    def shift_order_nrs(self, shift_amount = 0):
        
        if shift_amount == 0:
            return
        
        for idx in range(len(self.calib_data_dynamic)):
            order = self.calib_data_static[idx]
            order.order_nr += shift_amount
        
        
    def get_idx_by_order_nr(self, order_nr):
        for idx in range(len(self.calib_data_dynamic)):
            order = self.calib_data_static[idx]
            if order.order_nr == order_nr:
                return idx
        return None
    
    
    
    # Assumes that each order instance has been updated (avg_y is correct)
    def sort_orders(self, sort_plots = True):
        
        # Get indices to sort by
        y_values = [obj.avg_y for obj in self.calib_data_dynamic]
        sorted_idx = np.argsort(y_values) # top curves are low order nrs
        
        # Sort orders
        self.calib_data_dynamic = [self.calib_data_dynamic[idx] for idx in sorted_idx]
        
        # Sort corresponding plot objects if they're initialized
        if sort_plots:
            self.order_plot_points = [self.order_plot_points[idx] for idx in sorted_idx]
            self.order_poly_coefs = [self.order_poly_coefs[idx] for idx in sorted_idx]
            self.order_plot_curves = [self.order_plot_curves[idx] for idx in sorted_idx]
            self.order_bound_points = [self.order_bound_points[idx] for idx in sorted_idx]
        
        # Save order number in objects (done in static data)
        #for idx in range(len(self.calib_data_dynamic)):
        #    self.calib_data_dynamic[idx].order_nr = idx + self.first_order_nr
            
        # Re-select the correct order
        idxs = np.where(sorted_idx == self.selected_order_idx)[0] # get indices of rows where idx is the same, can be None
        self.selected_order_idx = None if len(idxs) == 0 else idxs[0]
        
        
        #self.reinitialize_orders() # TODO
    
    
   
    # Get z-values of corresponding x and y values and cut them according to left and right bounds
    def compile_full_spectrum(self):
        x_values = np.concatenate(self.orders_wavelengths[::-1]) # reverse order
        z_values = np.concatenate(self.orders_intensities[::-1]) # reverse order
        return x_values, z_values
    
    # Iterate over orders and save the order number as a complement to the x and z values
    def get_spectrum_order_nrs(self):
        order_nrs_list = []
        for order_idx in range(len(self.calib_data_dynamic) - 1, -1, -1): # reverse order
            order_nr = self.calib_data_static[order_idx].order_nr
            order_nrs_list.append(np.full(len(self.orders_wavelengths[order_idx]), order_nr))
        
        order_nrs = np.concatenate(order_nrs_list)
        return order_nrs
    
    # Iterate over all orders and update/initialize the data associated with the spectrum (wl and int etc.)
    def update_spectrum_data(self):
        for order_idx in range(len(self.calib_data_dynamic)):
            self.update_order_spectrum_data(order_idx)
            
    
    # Update/initialize the data associated with the spectrum (wl and int etc.) for the given order
    def update_order_spectrum_data(self, order_idx):
        # Get x_pixel values between bounds (e.g. array of ints from 480 to 701)
        self.update_spectrum_data_list(order_idx, self.orders_x_pixels, self.calc_order_x_pixels)
        
        # Calculate coefficients which are used to convert px => wavelength
        self.update_spectrum_data_list(order_idx, self.orders_wavelength_coefficients, self.calc_order_wavelength_coefs)
        
        # Get wavelengths corresponding to the x_pixels
        self.update_spectrum_data_list(order_idx, self.orders_wavelengths, self.calc_order_wavelengths)
        
        # Get intensities (integral over few px vertically) corresponding to the x_pixels
        self.update_spectrum_data_list(order_idx, self.orders_intensities, self.calc_order_intensities)
    
    
    # Either initialize or update spectrum data lists (values_fn output goes into array)
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
        [x_left, x_right] = self.calib_data_static[order_idx].bounds_px
        
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
        [px_start, px_end] = self.calib_data_static[order_idx].bounds_px
        [wave_start, wave_end] = self.calib_data_static[order_idx].bounds_wave
        [px_middle, wave_middle] = self.calib_data_static[order_idx].bounds_middle
        
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
    
    # Calculates the wavelengths of the pixels on a given order with the coefficients (no matter if linear or quadratic).
    def calc_order_intensities(self, order_idx):
        if order_idx > len(self.calib_data_dynamic):
            raise Exception(f'calc_order_wavelengths() | order_idx out of bounds: {order_idx}')
        
        # Get radius over which to integrate given order
        radius = self.get_order_radius(order_idx)
        
        # Get coordinates of the center pixels 
        x_array = self.orders_x_pixels[order_idx]
        y_array = np.polynomial.polynomial.polyval(x_array, self.order_poly_coefs[order_idx])
        #y_array = np.round(y_array).astype(int) # convert to same format as x-values
        
        
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
        #width = round(width)
        
        # Clip coordinates to be within bounds of the image
        x_array = np.clip(x_array, 0, self.max_x_idx).astype(int) 
        y_array = np.clip(y_array, 0, self.max_y_idx).astype(float)
        
        # Return the intensities from the diffr. order curve
        if radius == 0:
            return self.photo_array[y_array, x_array] # x and y have to be switched (first dimension is vertical on image)
        
        '''
        integrals = []
        for idx, x in enumerate(x_array):
            y_center = y_array[idx]
            integral = self.interpolate_integral_data(x, y_center, radius)
            integrals.append(integral)
        '''
        integrals = self.integrate_order_width_interpolated(x_array, y_array, radius)
        
        return integrals
    
    '''
    # The x_array are integers. y_array are floats but are bound by image top and bottom indices (0 and 1023).
    # Between the bounds the pixel value (z-value, intensity) is interpolation (quadratic fn) between neighboring pixels (vertical axis).
    def interpolate_integral_data(self, x, y, radius):
        
        # Get float y-index bounds for the integral
        high_y = y + radius
        low_y = y - radius
        high_y = clip(high_y, min_v = 0, max_v = self.max_y_idx)
        low_y = clip(low_y, min_v = 0, max_v = self.max_y_idx)
        
        # Direct sum between full pixel indices
        sum_high = math.floor(high_y)
        sum_low = math.ceil(low_y)
        sum_y_indices = np.arange(sum_low, sum_high + 1, dtype = int)
        sum_x_indices = np.full(sum_y_indices.shape, x, dtype = int)
        sum_array = self.photo_array[sum_y_indices, sum_x_indices] # Gather the pixel values from photo_array
        integral = np.sum(sum_array)
        
        # Get the next polynomial interpolation indices for fractional addition 
        interp_lowest = sum_low - 1
        interp_highest = sum_high + 1
        
        # Keep polynomial indices within image bounds
        offset_low = 0
        offset_high = 0
        if interp_lowest < 0:
            offset_low = 1
        elif interp_highest > self.max_y_idx: # radius isn't big enough to consider both simultaneously
            offset_high = -1
        
        
        # Get y-indices to do quadratic fn interpolation on
        x_indices = np.full((3, ), x, dtype = int)
        interpolation_indices_low = np.array([sum_low - 1, sum_low, sum_low + 1]) + offset_low
        interpolation_indices_high = np.array([sum_high - 1, sum_high, sum_high + 1]) + offset_high
        z_values_low = self.photo_array[interpolation_indices_low, x_indices]
        z_values_high = self.photo_array[interpolation_indices_high, x_indices]
        
        # Get value from fractional index. Do interpolation and multiply the value with remaining index fraction (smooth <=> discrete integral conversion).
        low_fraction = sum_low - low_y
        high_fraction = high_y - sum_high
        integral += simple_quadratic_interpolate(low_y, interpolation_indices_low, z_values_low) * low_fraction
        integral += simple_quadratic_interpolate(high_y, interpolation_indices_high, z_values_high) * high_fraction
        return integral
    '''
    
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
        needs_low_interp_mask = low_fraction > 1e-3 # Avoid calculations for zero fractions

        if np.any(needs_low_interp_mask):
            # Filter data for points needing low interpolation
            ylf_clip_subset = y_bounds_low_float[needs_low_interp_mask]
            x_coords_subset_low = x_indices[needs_low_interp_mask]

            # Determine stencil center: integer y pixel "closest" to the float boundary
            # Clip stencil center so that stencil_center +/- 1 are valid indices
            y_center_stencil_low = np.round(ylf_clip_subset).astype(int)
            y_center_stencil_low = np.clip(y_center_stencil_low, 1, self.max_y_idx - 1)
            
            yl_m1 = y_center_stencil_low - 1
            yl_0  = y_center_stencil_low
            yl_p1 = y_center_stencil_low + 1
            
            # Fetch z-values for the stencil points
            zl_m1 = self.photo_array[yl_m1, x_coords_subset_low]
            zl_0  = self.photo_array[yl_0,  x_coords_subset_low]
            zl_p1 = self.photo_array[yl_p1, x_coords_subset_low]
            
            val_at_low_bound = self.vectorized_quadratic_interpolate(
                ylf_clip_subset, y_center_stencil_low, zl_m1, zl_0, zl_p1
            )
            interp_contrib_low[needs_low_interp_mask] = val_at_low_bound * low_fraction[needs_low_interp_mask]

        # 7. Quadratic interpolation for the upper fractional part (only if high_fraction > 0)
        needs_high_interp_mask = high_fraction > 1e-3

        if np.any(needs_high_interp_mask):
            # Filter data for points needing high interpolation
            yhf_clip_subset = y_bounds_high_float[needs_high_interp_mask]
            x_coords_subset_high = x_indices[needs_high_interp_mask]

            y_center_stencil_high = np.round(yhf_clip_subset).astype(int)
            y_center_stencil_high = np.clip(y_center_stencil_high, 1, self.max_y_idx - 1)

            yh_m1 = y_center_stencil_high - 1
            yh_0  = y_center_stencil_high
            yh_p1 = y_center_stencil_high + 1
            
            zh_m1 = self.photo_array[yh_m1, x_coords_subset_high]
            zh_0  = self.photo_array[yh_0,  x_coords_subset_high]
            zh_p1 = self.photo_array[yh_p1, x_coords_subset_high]

            val_at_high_bound = self.vectorized_quadratic_interpolate(
                yhf_clip_subset, y_center_stencil_high, zh_m1, zh_0, zh_p1
            )
            interp_contrib_high[needs_high_interp_mask] = val_at_high_bound * high_fraction[needs_high_interp_mask]
            
        # 8. Combine all parts: Bulk sum + contribution from lower fraction + contribution from upper fraction
        total_integrals = bulk_sum_values + interp_contrib_low + interp_contrib_high
        
        return total_integrals
    
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
    
    
    # Check if top orders are too close to the edge and if so then don't use these
    def check_top_orders_proximity(self):
        something_changed = False
        
        max_idx = min(10 + 1, len(self.calib_data_dynamic)) # no point in checking all orders (ignore bottom)
        for order_idx in range(max_idx):
            
            if self.ignore_top_orders and self.if_needs_cutting(order_idx):
                if self.calib_data_static[order_idx].use_order:
                    something_changed = True 
                
                self.calib_data_static[order_idx].use_order = False
                color = 'tab:gray'
            else:
                if not self.calib_data_static[order_idx].use_order:
                    something_changed = True 
                
                self.calib_data_static[order_idx].use_order = True
                color = 'white' if order_idx == self.selected_order_idx else 'r'
                
            self.update_one_order_curve_color(order_idx, set_color = color)
            
            if something_changed:
                self.update_spectrum(force_scale = True)
            
            # TODO for v3: think through the cases of different length input bounds
            
    
    # Check if order is too close to top edge of image
    def if_needs_cutting(self, order_idx):
        # Get topmost point's y-coordinate
        top = min(self.calib_data_dynamic[order_idx].ylist) # min because coordinate increases towards bottom
        
        # Get radius over which to integrate given order
        radius = self.get_order_radius(order_idx)
        
        # If pixel value is lower because of image clipping then the order needs to be cut
        if top - radius < 0:
            return True
        else:
            return False
            
        
    
    '''
    def point_in_range(self, x_point, x1, y1, x2, y2, x3, y3):
        
        x_start,_ = intersection(x1,y1,x2,y2) # intersection function is way too slow
        x_start = x_start[0]
        
        x_end,_ = intersection(x1,y1,x3,y3)
        x_end = x_end[0]
        
        if x_start <= x_point <= x_end:
            return True
        return False
    '''

# Do linear regression and find the y-value at provided x
def linear_regression(x, x_start, x_end, y_start, y_end):
    
    dx = x_end - x_start
    dy = y_end - y_start
    slope = dy / dx
    intercept = y_start - slope * x_start
    
    y = x * slope + intercept
    return y

# Calculate quadratic fn and then fint the value of y at x
def quadratic_regression(x, x_values, y_values):
    poly_coefs = np.polynomial.polynomial.polyfit(x_values, y_values, 2)
    y = poly_coefs[2] * x ** 2 + poly_coefs[1] * x + poly_coefs[0]
    return y
'''
# Sum pixels around the order
# If the width is even number then take asymmetrically one pixel from lower index
# if use_weights is True then the integral is summed with Gaussian weights (max is in center) with FWHM of width
def integrate_order_width(photo_array, x_pixel, y_pixel, width = 1, use_weights = False):
    width = round(width)
    
    if width == 1:
        return photo_array[y_pixel, x_pixel] # x and y have to be switched (somewhy)
    
    integral = 0
    x_pixel = clip(x_pixel, 0, self.max_x_idx)
    idx_lower = math.floor(width / 2)
    idx_higher = math.ceil(width / 2) # range omits last value
    for y_idx in range(y_pixel - idx_lower, y_pixel + idx_higher):
        y_idx = clip(y_idx, 0, self.max_y_idx)
        
        gaussian_weight = 1
        if use_weights:
            a = 1 # normalized
            c = width / 2.35482 # width is FWHM
            gaussian_weight = a * math.exp(-(y_pixel - y_idx) ** 2 / 2 / c ** 2)
        
        integral += photo_array[y_idx, x_pixel] * gaussian_weight # x and y have to be switched (somewhy)
    
    return integral
'''

def get_polynomial_intersection(coefs1, coefs2, image_size, is_left = True):
    x1, y1, x2, y2 = calc_polynomial_intersection(coefs1, coefs2)
    best_x = None
    
    # check if values are in bounds of the image
    if (0 <= x1 <= image_size - 1) and (0 <= y1 <= image_size - 1):
        best_x = x1
    
    if (0 <= x2 <= image_size - 1) and (0 <= y2 <= image_size - 1):
        
        if best_x is None: # only second point is ok
            return x2
        else: # both points are ok, select best
            
            if is_left:
                best_x = max(x1, x2)
            else:
                best_x = min(x1, x2)
            return best_x
    
    # 2nd point was no good
    return best_x

# assumes 2nd order
def calc_polynomial_intersection(coefs1, coefs2):
    a = coefs1[2] - coefs2[2]
    b = coefs1[1] - coefs2[1]
    c = coefs1[0] - coefs2[0]
    
    
    sqrt_val = b ** 2 - 4 * a * c
    x1 = (-b + np.sqrt(sqrt_val)) / 2 / a
    x2 = (-b - np.sqrt(sqrt_val)) / 2 / a
    
    y1 = coefs1[2] * x1 ** 2 + coefs1[1] * x1 + coefs1[0]
    y2 = coefs1[2] * x2 ** 2 + coefs1[1] * x2 + coefs1[0]
    
    return x1, y1, x2, y2


# sort the arrays in increasing value or first array
def sort_related_arrays(arr1, arr2_list):
    sort_idx = np.argsort(arr1)
    arr1 = arr1[sort_idx]
    if type(arr2_list) == 'list':
        for idx in range(len(arr2_list)):
            arr2_list[idx] = arr2_list[idx][sort_idx]
    else: # not list but single array
        arr2_list = arr2_list[sort_idx]
    return arr1, arr2_list

# Create a combined list of points
def gather_points(list_of_lists):
    combined_list = []
    for sub_list in list_of_lists:
        combined_list += sub_list
    return combined_list

# Calculate points of polynomial for plotting
def get_polynomial_points(order, arr_length, flip = False):
    # get regression coeficients
    poly_coefs = np.polynomial.polynomial.polyfit(order.xlist, order.ylist, 2)
    
    if flip: # treat x as y and y as x
        use_coefs = np.polynomial.polynomial.polyfit(order.ylist, order.xlist, 2)
    else: # normal
        use_coefs = poly_coefs
    
    # get array of points on image for the polynomial
    curve_array = np.empty(arr_length)
    for idx in range(arr_length):
        curve_array[idx] = poly_func_value(idx, use_coefs)
    
    
    # clip the values
    curve_array = np.clip(curve_array, 0, arr_length - 1)
    
    return curve_array, poly_coefs

# Take the x and y values and get quadratic fn for them. Then calculate the y at x.
def simple_quadratic_interpolate(x, x_array, y_array):
    '''
    from scipy.interpolate import interp1d
    
    # Create linear interpolator with extrapolation enabled
    linear_interp = interp1d(x_array, y_array, kind='quadratic', fill_value='extrapolate', assume_sorted=False)
    
    # Interpolate multipliers at the spectrum wavelengths
    y = linear_interp(x)
    '''
    coefs = np.polynomial.polynomial.polyfit(x_array, y_array, 2)
    y = np.polynomial.polynomial.polyval(x, coefs)
    return y


##############################################################################################################
# Utility functions
##############################################################################################################

# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
# Either returns all files in folder or returns files that match the series_filename string where _0001 number may change
# series_filename HAS to be in the format 'whatever_0001.tif' or 'whatever_0001_abc.tif'
# That is, '_0001' exact substring HAS to be in the series_filename string (this has to be hardcoded)
# If series_filename is specified then code tries to find the files with the sample name, otherwise takes first name 
# (best result is if only one sample series is in folder)
def get_folder_files(path, series_filename = None, return_all = False):
    
    # There's both identificator_start and identificator_end because sometimes _0001 is in the middle of the file name
    identificator_start = None
    identificator_end = None
    
    # Get files, not directories
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    onlyfiles.sort() # sort filenames, so _0001 is before _0002
    
    
    # Extract all files
    if return_all:
        return onlyfiles
    
    # Extract only files that match the (identificator_start + '_0001' + identificator_end + '.tif')
    else:
        
        # This part is messy because of broken naming convention from experiments at JET.
        # I have to harcode _0001 here because there's e.g. 
        # '14IWG1A_11a_P11_gate_1000ns_delay_1000ns_0001.tif' and
        # '492_2X8_R7C3C7_0001 2X08 - R7 C3-C7.tif' in the files
        
        # take the first filename, this should contain '_0001' after sorting
        if series_filename is None:
            name = onlyfiles[0]
        else:
            name = series_filename
        
        # get file identificator start and end
        digits_pattern = r'^(.*?)_0001(.*?)$' # I have to hardcode _0001 like this because naming convention at JET failed
        regex_result = re.search(digits_pattern, name)
        identificator_start = regex_result[1]
        identificator_end = regex_result[2]
        
        # Escape the special regex characters since they are used as a pattern
        identificator_start = re.escape(identificator_start)
        identificator_end = re.escape(identificator_end)
        
        
        # Compile the pattern (even if _0001 is in the middle of the name)
        pattern = '^' + identificator_start + r'_\d+?' + identificator_end + '$'
        
        output_files = [f for f in onlyfiles if re.search(pattern, f) ]
    
    return output_files

# Create output folder if it doesn't exist
def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

# Check if string can be converted to number
def str_is_number(string):
    try: # convert to float but errors and nil and empty field is None
        float(string)
        return True
    except ValueError:
        return False
    except TypeError:
        return False

# Return a value for the polynomial
def poly_func_value(x_value, coefs):
    value = 0
    for idx, coef in enumerate(coefs):
        value += coef * x_value ** idx
    return round(value, 10)

# Return the distance between two xy points
def xy_distance(point1x, point1y, point2x, point2y):
    return np.sqrt((point2x - point1x) ** 2 + (point2y - point1y) ** 2)

def clip(value, min_v = -math.inf, max_v = math.inf):
    return max(min_v, min(value, max_v))


############################################################
# Sukhbinder
# 5 April 2017
# https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
############################################################
def _rect_inter_inner(x1,x2):
    n1=x1.shape[0]-1
    n2=x2.shape[0]-1
    X1=np.c_[x1[:-1],x1[1:]]
    X2=np.c_[x2[:-1],x2[1:]]    
    S1=np.tile(X1.min(axis=1),(n2,1)).T
    S2=np.tile(X2.max(axis=1),(n1,1))
    S3=np.tile(X1.max(axis=1),(n2,1)).T
    S4=np.tile(X2.min(axis=1),(n1,1))
    return S1,S2,S3,S4

def _rectangle_intersection_(x1,y1,x2,y2):
    S1,S2,S3,S4=_rect_inter_inner(x1,x2)
    S5,S6,S7,S8=_rect_inter_inner(y1,y2)

    C1=np.less_equal(S1,S2)
    C2=np.greater_equal(S3,S4)
    C3=np.less_equal(S5,S6)
    C4=np.greater_equal(S7,S8)

    ii,jj=np.nonzero(C1 & C2 & C3 & C4)
    return ii,jj

def intersection(x1,y1,x2,y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
usage:
x,y=intersection(x1,y1,x2,y2)
    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)
    x2=phi    
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)
    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()
    """
    ii,jj=_rectangle_intersection_(x1,y1,x2,y2)
    n=len(ii)

    dxy1=np.diff(np.c_[x1,y1],axis=0)
    dxy2=np.diff(np.c_[x2,y2],axis=0)

    T=np.zeros((4,n))
    AA=np.zeros((4,4,n))
    AA[0:2,2,:]=-1
    AA[2:4,3,:]=-1
    AA[0::2,0,:]=dxy1[ii,:].T
    AA[1::2,1,:]=dxy2[jj,:].T

    BB=np.zeros((4,n))
    BB[0,:]=-x1[ii].ravel()
    BB[1,:]=-x2[jj].ravel()
    BB[2,:]=-y1[ii].ravel()
    BB[3,:]=-y2[jj].ravel()

    for i in range(n):
        try:
            T[:,i]=np.linalg.solve(AA[:,:,i],BB[:,i])
        except:
            T[:,i]=np.NaN


    in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

    xy0=T[2:,in_range]
    xy0=xy0.T
    return xy0[:,0],xy0[:,1]


##############################################################################################################
# Debugging
##############################################################################################################


# Automatically wrap all functions in the current module (i.e., script)
if dbg:
    wrap_functions_in_module()
    calibration_window = count_calls(calibration_window)
        


##############################################################################################################
# Vertical bounds and wavelengths data
##############################################################################################################

# This is a constant array in .aryx files (in aif subfile in aryx compressed file).
# The array is extracted with the code from 
# https://gitlab.com/ltb_berlin/ltb_files and
# https://ltb_berlin.gitlab.io/ltb_files/ltbfiles.html

'''
# The data encoding in aif file is the following
AIF_DTYPE_ARYX = numpy.dtype([('indLow', numpy.int32),
                      ('indHigh', numpy.int32),
                      ('order', numpy.int16),
                      ('lowPix', numpy.int16),
                      ('highPix', numpy.int16),
                      ('foo', numpy.int16),
                      ('lowWave', numpy.float64),
                      ('highWave', numpy.float64)]) 
'''

# The data is extracted from sample 492 (JET experiment). The data is constant throughout the experiments.
# Only the smallest order nr row is cut from e.g. 2024.08.12 calibration day spectra because the order was
# too close to the image edge.
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


# This array is gained by fitting quadratic fn to every order in Sophi nXt output spectrum 
# for sample 492 (perfect fit).
#order_nr,a,b,c
wavelength_calculation_coefficients = np.array([
[35,-7.00E-07,0.026221,744.659717],
[36,-6.76E-07,0.025487,723.968536],
[37,-6.57E-07,0.024796,704.396709],
[38,-6.40E-07,0.024142,685.855902],
[39,-6.23E-07,0.023523,668.266578],
[40,-6.12E-07,0.022938,651.55623],
[41,-5.98E-07,0.022378,635.662755],
[42,-5.80E-07,0.021842,620.52635],
[43,-5.67E-07,0.021334,606.093614],
[44,-5.54E-07,0.02085,592.317075],
[45,-5.43E-07,0.020387,579.152704],
[46,-5.31E-07,0.019944,566.560956],
[47,-5.19E-07,0.019519,554.50497],
[48,-5.08E-07,0.019113,542.95104],
[49,-4.99E-07,0.018723,531.86857],
[50,-4.89E-07,0.018349,521.229365],
[51,-4.79E-07,0.01799,511.007244],
[52,-4.71E-07,0.017645,501.178152],
[53,-4.62E-07,0.017312,491.719804],
[54,-4.54E-07,0.016992,482.611633],
[55,-4.45E-07,0.016683,473.8348],
[56,-4.36E-07,0.016385,465.37118],
[57,-4.29E-07,0.016098,457.204276],
[58,-4.22E-07,0.01582,449.318976],
[59,-4.14E-07,0.015552,441.700825],
[60,-4.07E-07,0.015294,434.336412],
[61,-4.01E-07,0.015043,427.213362],
[62,-3.95E-07,0.014801,420.319964],
[63,-3.88E-07,0.014566,413.645358],
[64,-3.82E-07,0.014339,407.179242],
[65,-3.76E-07,0.014118,400.911928],
[66,-3.71E-07,0.013905,394.83442],
[67,-3.65E-07,0.013698,388.938243],
[68,-3.60E-07,0.013496,383.215419],
[69,-3.54E-07,0.013301,377.658416],
[70,-3.49E-07,0.013111,372.260075],
[71,-3.44E-07,0.012927,367.013682],
[72,-3.40E-07,0.012747,361.912971],
[73,-3.35E-07,0.012573,356.951954],
[74,-3.31E-07,0.012403,352.124949],
[75,-3.26E-07,0.012238,347.426573],
[76,-3.22E-07,0.012077,342.851777],
[77,-3.18E-07,0.01192,338.395757],
[78,-3.14E-07,0.011768,334.053939],
[79,-3.10E-07,0.011619,329.821965],
[80,-3.06E-07,0.011474,325.695765],
[81,-3.02E-07,0.011332,321.671399],
[82,-2.99E-07,0.011195,317.74509],
[83,-2.95E-07,0.01106,313.913386],
[84,-2.92E-07,0.010928,310.172899],
[85,-2.88E-07,0.0108,306.520343],
[86,-2.85E-07,0.010674,302.952695],
[87,-2.82E-07,0.010552,299.46701],
[88,-2.79E-07,0.010432,296.06053],
[89,-2.75E-07,0.010315,292.730567],
[90,-2.72E-07,0.010201,289.474537],
[91,-2.69E-07,0.010089,286.290057],
[92,-2.66E-07,0.009979,283.174805],
[93,-2.64E-07,0.009872,280.126503],
[94,-2.61E-07,0.009767,277.143023],
[95,-2.58E-07,0.009664,274.222291],
[96,-2.56E-07,0.009564,271.362386],
[97,-2.53E-07,0.009465,268.561468],
[98,-2.50E-07,0.009369,265.817678],
[99,-2.48E-07,0.009274,263.129291],
[100,-2.45E-07,0.009182,260.494646],
[101,-2.43E-07,0.009091,257.912157],
[102,-2.40E-07,0.009002,255.380284],
[103,-2.38E-07,0.008914,252.897539],
[104,-2.36E-07,0.008829,250.462526],
[105,-2.34E-07,0.008745,248.073874],
[106,-2.32E-07,0.008662,245.730265],
[107,-2.29E-07,0.008582,243.430441],
[108,-2.27E-07,0.008502,241.173221],
[109,-2.25E-07,0.008424,238.957405],
[110,-2.23E-07,0.008348,236.781855],
[111,-2.21E-07,0.008272,234.64548],
[112,-2.19E-07,0.008199,232.547217],
[113,-2.17E-07,0.008126,230.486076],
[114,-2.15E-07,0.008055,228.461106],
[115,-2.14E-07,0.007985,226.47135],
[116,-2.12E-07,0.007916,224.515874],
[117,-2.10E-07,0.007849,222.593802],
[118,-2.08E-07,0.007782,220.704322],
[119,-2.07E-07,0.007717,218.846563],
[120,-2.05E-07,0.007653,217.019765],
[121,-2.03E-07,0.00759,215.223181],
[122,-2.01E-07,0.007527,213.456005],
[123,-2.00E-07,0.007466,211.717568],
[124,-1.98E-07,0.007406,210.007169],
[125,-1.97E-07,0.007347,208.324113],
[126,-1.95E-07,0.007289,206.667762],
[127,-1.94E-07,0.007231,205.037474],
[128,-1.92E-07,0.007175,203.432694],
[129,-1.91E-07,0.007119,201.852738],
[130,-1.90E-07,0.007065,200.296964],
[131,-1.85E-07,0.007008,198.765832]
])


# TODO: detect and cut top order if close to edge like when _Vertical_points.csv doesn't have the line

# order_nr, start px, end px, start wl, end wl, a, b, c
orders_info = np.concatenate((extracted_orders_info[:, [2,3,4,6,7]], wavelength_calculation_coefficients[:,[1,2,3]]), axis=1)

# These values are gained by analyzing the spectrum from Sophi nXt with sample 492.
# Wavelengths of other spectra can be calculated with 1 pm accuracy with horizontal shift.
def order_px_to_wavelength(order_nr, px, horizontal_shift = 0):
    
    # Each order and its wavelengths follow a quadratic fn y = a * x^2 + b * x + c
    # px in the equation is the image px, not order px. Px is from 0 to 1023 and bounds don't matter.
    # These coefficients in turn follow a very strong (smallest R^2 was 0.99997) linear (division) curve.
    #a = -0.0000243121 / order_nr - 0.0000000022
    #b = 0.9169981497 / order_nr + 0.0000110699
    #c = 26072.6114478297 / order_nr - 0.2305172110
    
    # get index of row where order_nr is the same
    idxs = np.where(order_nr == orders_info[:,0])[0]
    if len(idxs) == 0: 
        print('ERROR: order_px_to_wavelength() | no match for order ' + str(order_nr))
        return
    else: 
        idx = idxs[0]
    
    # The direct polynomial coefficients still give more accurate result than a,b,c functions.
    a = orders_info[idx, 5]
    b = orders_info[idx, 6]
    c = orders_info[idx, 7]
    
    px += horizontal_shift # TODO: check shift sign
    return a * px^2 + b * px + c

##############################################################################################################
# RUN MAIN PROGRAM
##############################################################################################################
main_program()