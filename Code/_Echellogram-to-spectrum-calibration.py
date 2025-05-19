# Author: Jasper Ristkok
# v2.3

# Code to convert an echellogram (photo) to spectrum

# TODO: check out of bounds orders
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
        self.bounds_px_original = [None, None] # pixel index from input file
        self.bounds_wave_original = [None, None] # wavelengths from input file
        self.use_order = True
        
        if existing_data:
            self.load_encode(existing_data)
    
    # Convert points into arrays for saving
    def save_decode(self):
        static_dict = {}
        static_dict['order_nr'] = self.order_nr
        static_dict['bounds_px'] = self.bounds_px
        static_dict['bounds_wave'] = self.bounds_wave
        static_dict['bounds_px_original'] = self.bounds_px_original
        static_dict['bounds_wave_original'] = self.bounds_wave_original
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
        self.best_order_idx = None
        self.autoscale_spectrum = True
        self.shift_wavelengths = True # If self.shift_wavelengths == True then wavelengths are locked to order curves, otherwise to pixels on image
        self.curve_edit_mode = False # TODO: delete?
        
        
        self.photo_array = [] # assumes square image
        self.bounds_input_array = []
        self.order_plot_curves = []
        self.order_plot_points = []
        self.order_poly_coefs = []
        self.order_bound_points = []
        
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
        
        btn_autocalibrate.pack()
        btn_button_save.pack()
        btn_ignore_orders.pack()
        
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
            
            self.first_order_nr = int(self.bounds_input_array[:, 0].min()) # assumes that input file has correct data
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
    def compile_static_data_from_file(self):
        # reset the list, only to contain static data that is relevant for drawn orders
        self.calib_data_static = []
        
        input_array = self.bounds_input_array
        if input_array is None:
            return
        
        dynamic_len = len(self.calib_data_dynamic) # nr of drawn curves
        static_len = input_array.shape[0] # nr of orders in input bounds file
        
        # Let user know that input data wasn't perfect
        if dynamic_len < static_len:
            print('_Vertical_points.csv has more rows than drawn orders. Highest order numbers in the file are ignored.')
            self.set_feedback('_Vertical_points.csv has more rows than drawn orders. Highest order nrs are ignored.', 15000)
        
        # Get row where order nr is same as self.first_order_nr
        idxs = np.where(input_array[:,0] == self.first_order_nr)[0]
        if len(idxs) > 0: # There's match
            idx = idxs[0]
        else:
            print('First order nr not in _Vertical_points.csv')
            #self.set_feedback('First order nr not in _Vertical_points.csv', 15000)
            raise Exception('First order nr not in _Vertical_points.csv')
            
        
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
            px_start = input_array[row_idx, 1]
            px_end = input_array[row_idx, 2]
            
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
            
            static_class_instance.order_nr = order_nr
            static_class_instance.bounds_px = [px_start, px_end]
            static_class_instance.bounds_px_original = [px_start, px_end]
            
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
                self.calib_data_static.append(static_class_instance)
        
        # In case there has been a shift, recalculate shifted bounds from original bounds
        for order_idx in range(len(self.calib_data_static)):
            self.recalculate_bounds(order_idx)
    
    
    # Save calibration curves and the corresponding points
    def save_data(self):
        
        # decode objects into dictionaries
        save_dict = {'dynamic' : [], 'static' : []}
        
        
        # Iterate over orders
        for order_idx, order in enumerate(self.calib_data_dynamic):
            if self.ignore_top_orders and self.if_needs_cutting(order_idx):
                continue # TODO for v3: change when implementing edge cutting
            
            points_array = order.save_decode()
            save_dict['dynamic'].append(points_array)
        
        for order in self.calib_data_static:
            static_dict = order.save_decode()
            save_dict['static'].append(static_dict)
        
        save_dict['first_order_nr'] = self.first_order_nr
        save_dict['total_shift_right'] = self.total_shift_right
        save_dict['total_shift_up'] = self.total_shift_up
        
        # Save as JSON readable output
        with open(output_path + '_Calibration_data.json', 'w') as file: # '_' in front to find easily among Echellograms
            json.dump(save_dict, file, sort_keys=True, indent=4)
        
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
            file_first_order_nr = int(self.bounds_input_array[:, 0].min())
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
    
    
    # Reset program mode
    def reset_mode(self, button_call = True):
        
        # Reset selected order color
        self.update_one_order_curve(self.best_order_idx, single_call = True, recalculate = False, set_color = 'red')
        
        self.program_mode = None
        self.best_order_idx = None
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
            
            for idx in range(len(self.calib_data_dynamic)):
                order_nr = self.calib_data_static[idx].order_nr
                coefs = self.order_poly_coefs[idx]
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
        x_values, z_values, order_nrs = self.get_order_spectrum()
        data = np.column_stack((x_values, z_values, order_nrs))
        np.savetxt(output_path + '_Spectrum.csv', data, fmt = '%.8e', delimiter = ',', comments = '', header = 'Wavelength (nm),Intensity,Order nr')
        
            
    
    
    # Overwrite the order points with the ones calculated from polynomials
    def orders_tidy(self):
        start_x = 25
        center_x = math.floor(self.photo_array.shape[0] / 2)
        end_x = self.photo_array.shape[0] - 25
        
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
        #self.initialize_bounds(self.best_order_idx)
        
        self.plot_order(diffr_order, color = 'white')
        
        # sort orders by average y values
        #self.calib_data_dynamic.sort(key=lambda x: x.avg_y, reverse=True)
        
        self.selected_order_nr = diffr_order
        self.best_order_idx = len(self.calib_data_dynamic) - 1
        
        
        self.draw_bounds(self.best_order_idx)
        
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
        
        if self.best_order_idx is None:
            self.set_feedback('No order selected', 8000)
            return
        
        # Show orders
        self.show_orders = True
        self.hide_unhide_orders()
        
        selected_idx = self.best_order_idx
        order_nr = self.calib_data_static[selected_idx].order_nr
        
        del self.calib_data_dynamic[selected_idx]
        
        # Recompile self.calib_data_static with relevant data
        self.compile_static_data_from_file()
        
        # Deselect the order
        self.selected_order_nr = None
        self.best_order_idx = None
        
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
        if self.best_order_idx == best_order_idx:
            
            self.best_order_idx = None
            self.selected_order_nr = None
            
            self.update_one_order_curve(best_order_idx, single_call = True, recalculate = False, set_color = 'red')
            self.set_feedback('Order ' + str(best_order_idx + self.first_order_nr) + ' deselected')
        
        # Select
        else:
            # paint last selection red
            self.update_one_order_curve(self.best_order_idx, single_call = True, recalculate = False, set_color = 'red')
            
            self.best_order_idx = best_order_idx
            self.selected_order_nr = self.calib_data_static[best_order_idx].order_nr
            
            self.update_one_order_curve(best_order_idx, single_call = True, recalculate = False, set_color = 'white')
            self.set_feedback('Order ' + str(best_order_idx + self.first_order_nr) + ' selected')
    
    
    def order_click(self, click_point):
        
        if self.selected_order_nr is None:
            self.set_feedback('Order mode but none selected')
            return
        
        if not self.show_orders:
            return
        
        order = self.calib_data_dynamic[self.best_order_idx]
        
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
        self.calib_data_dynamic[self.best_order_idx].points[best_point_idx] = click_point # TODO: delete hack 
        self.calib_data_dynamic[self.best_order_idx].update()
        
        
        # sort orders by average y values, this might change order nrs
        self.sort_orders()
        
        # Calculate the bounds for that order
        self.recalculate_bounds(self.best_order_idx)
        
        # TODO: update only the selected order and all bounds
        self.update_order_curves()
        
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
            self.update_spectrum(autoscale = True)
    
    def shift_orders_right(self, shift_amount = 0, button_call = True):
        if shift_amount == 0:
            return
        
        # Register total shift
        self.total_shift_right += shift_amount
        
        for order_idx in range(len(self.calib_data_dynamic)):
            order = self.calib_data_dynamic[order_idx]
            
            # shift order points
            for point in order.points:
                point.x += shift_amount # shift right
            
            self.recalculate_bounds(order_idx, last_shift = shift_amount)
            
        
        self.set_feedback('Orders shifted right by ' + str(shift_amount) + ' px')
        
        # update visuals
        if button_call:
            self.update_point_instances()
            self.update_order_curves()
            self.calculate_all_bounds(initialize_x = True)
            self.update_spectrum(autoscale = True)
    
    
    def recalculate_bounds(self, order_idx, last_shift = 0):
        
        # Short-circuit the function if there's no need to properly calculate stuff again
        if (self.total_shift_right == 0) and (last_shift == 0):
            return self.calib_data_static[order_idx].bounds_px
        
        # shift order bounds
        [orig_px_left, orig_px_right] = self.calib_data_static[order_idx].bounds_px_original
        left_px = clip(orig_px_left + self.total_shift_right, 0, self.photo_array.shape[0] - 1)
        right_px = clip(orig_px_right + self.total_shift_right, 0, self.photo_array.shape[0] - 1)
        self.calib_data_static[order_idx].bounds_px = [left_px, right_px]
        
        # If self.shift_wavelengths == True then wavelengths are locked to order curves, otherwise to pixels on image
        if not self.shift_wavelengths:
            [orig_wave_left, orig_wave_right] = self.calib_data_static[order_idx].bounds_wave_original
            
            left_wave = linear_regression(left_px, orig_px_left, orig_px_right, orig_wave_left, orig_wave_right)
            right_wave = linear_regression(right_px, orig_px_left, orig_px_right, orig_wave_left, orig_wave_right)
            self.calib_data_static[order_idx].bounds_wave = [left_wave, right_wave]
        
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
        x_values, z_values, order_nrs = self.get_order_spectrum()
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
        self.best_order_idx = None
        
        self.update_point_instances()
        
        # sort orders by average y values
        self.sort_orders()
        
        self.update_order_curves()
        
        
        self.calculate_all_bounds(initialize_x = True)
            
        
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
        
    
    def update_spectrum(self, autoscale = False):
         
         x_values, z_values, order_nrs = self.get_order_spectrum()
         
         # Plot curve
         self.spectrum_curve.set_xdata(x_values)
         self.spectrum_curve.set_ydata(z_values)
         
         # rescale to fit
         if self.autoscale_spectrum and autoscale:
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
        idxs = np.where(sorted_idx == self.best_order_idx)[0] # get indices of rows where idx is the same, can be None
        self.best_order_idx = None if len(idxs) == 0 else idxs[0]
        
        
        #self.reinitialize_orders() # TODO
    
    
   
    # Get z-values of corresponding x and y values and cut them according to left and right bounds
    def get_order_spectrum(self):
        x_values = None
        z_values = []
        order_nrs = []
        
        
        nr_pixels = self.photo_array.shape[0]
        photo_pixels = np.arange(nr_pixels)
        
        # Compile z-data from the photo with all diffraction orders
        # Iterate over orders backwards because they are sorted by order nr (top to bottom)
        for order_idx in range(len(self.calib_data_dynamic) - 1, -1, -1):
            order_nr = self.calib_data_static[order_idx].order_nr
            
            #curve_x = photo_pixels
            curve_x = self.order_plot_curves[order_idx].get_xdata()
            curve_y = self.order_plot_curves[order_idx].get_ydata()
            curve_y = np.round(curve_y)
            
            # sort the arrays in increasing x value
            #curve_x, curve_y = sort_related_arrays(curve_x, curve_y)
            
            
            # Interpolate and get y-values of the calibration curve at every pixel
            #y_pixels = np.interp(photo_pixels, curve_x, curve_y) # pixel values are already calculated by the polynomial
            #y_pixels = y_pixels.astype(np.int32) # convert to same format as x-values
            y_pixels = curve_y.astype(np.int32) # convert to same format as x-values
            
            
            # Get bounds for this diffraction order
            [x_left, x_right] = self.calib_data_static[order_idx].bounds_px
            
            # Default bounds as first px and last px
            if x_left is None:
                x_left = 0
            if x_right is None:
                x_right = nr_pixels - 1
            
            # get z-values corresponding to the interpolated pixels on the order curve
            # Save only pixels that are between bounds, check 20 px left and right from bound for intersection
            for idx2 in np.arange(max(0, x_left - 20), min(nr_pixels, x_right + 1 + 20), dtype = int):
                
                # Save px only if it's between left and right bounds
                x_px = curve_x[idx2]
                
                # Do stuff if point is between bounds
                #if self.point_in_range(x_value, curve_x, curve_y, x2, y2, x3, y3): # intersection function is way too slow
                if x_left <= x_px <= x_right:
                    order_nrs.append(order_nr)
                    
                    y_px = y_pixels[idx2]
                    
                    [wave_start, wave_end] = self.calib_data_static[order_idx].bounds_wave
                    
                    # No input with wavelengths, output x-axis as pixel values
                    if wave_start is None:
                        if x_values is None: # Initialize
                            x_values = [0]
                        else:
                            x_values.append(x_values[-1] + 1) # increment by 1
                        
                        
                    # Has input with wavelengths, output x-axis as wavelengths
                    else:
                        if x_values is None: # Initialize
                            x_values = []
                        
                        x_value = self.get_wavelength(order_idx, x_px)
                        x_values.append(x_value)
                        
                    
                    # Get the width to integrate over (between two diffraction orders)
                    width = 1 #self.integral_width
                    center = self.calib_data_dynamic[order_idx].avg_y
                    if order_idx > 0: # check for out of bounds error
                        low = self.calib_data_dynamic[order_idx - 1].avg_y
                        width = (center - low) / 2
                    elif len(self.calib_data_dynamic) > order_idx + 1: # check for out of bounds error
                        high = self.calib_data_dynamic[order_idx + 1].avg_y
                        width = (high - center) / 2
                    
                    # very important, otherwise 2.49 and 2.51 will have 150% jump in integral because of rounding
                    width = clip(width, min_v = 3) 
                    
                    
                    # Get z value from Echellogram
                    integral = integrate_order_width(self.photo_array, x_px, y_px, width = width)
                    z_values.append(integral) # x and y have to be switched (somewhy)
                    
        # While at it, save spectrum total intensity
        self.spectrum_total_intensity = sum(z_values)
        
        return x_values, z_values, order_nrs
    
    

    def get_wavelength(self, order_idx, x_px):
        
        # Take bounds from (latest aka shifted) order points class
        [px_start, px_end] = self.calib_data_static[order_idx].bounds_px
        [wave_start, wave_end] = self.calib_data_static[order_idx].bounds_wave
        
        # Do linear regression and find the wavelength of x_px
        wavelength = linear_regression(x_px, px_start, px_end, wave_start, wave_end)
        return wavelength
    
    # Check if top orders are too close to the edge and if so then don't use these
    def check_top_orders_proximity(self):
        
        max_idx = min(20 + 1, len(self.calib_data_dynamic)) # no point in checking all orders (ignore bottom)
        for order_idx in range(max_idx):
            
            if self.ignore_top_orders and self.if_needs_cutting(order_idx):
                self.calib_data_static[order_idx].use_order = False
            else:
                self.calib_data_static[order_idx].use_order = True
            
            # TODO for v3: think through the cases of different length input bounds and 
            # implement edge cutting
    
    # Check if order is too close to top edge of image
    def if_needs_cutting(self, order_idx):
        # Get topmost point's y-coordinate
        top = min(self.calib_data_dynamic[order_idx].ylist) # min because coordinate increases towards bottom
        if order_idx >= len(self.calib_data_dynamic) - 1: # last order (bottom)
            second = min(self.calib_data_dynamic[order_idx - 1].ylist)
        else: # take next order
            second = min(self.calib_data_dynamic[order_idx + 1].ylist)
        dy = abs(second - top)
        
        # If pixel value is lower because of image clipping then the order needs to be cut
        if top < dy / 2:
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

def draw_pyplot(photo_array, root_fig = None, identificator = ''):
    #%matplotlib
    
    if root_fig is None:
        fig = plt.figure(figsize=[10,5])
    else:
        fig = root_fig
    
    
    #plt.plot(photo_array)
    
    #plt.imshow( photo_array )
    plt.title(identificator)
    
    #img = plt.imshow(photo_array, norm = 'log')
    
    # Initializes figure   
    #fig = plt.figure(figsize=[10,5])
    #ax = fig.add_subplot(1,1,1)
    
    
    
   
    
    #heatmap=ax.contourf(photo_array)
    #plt.imshow(photo_array, cmap='hot', interpolation='nearest')
    
#    handle, = ax.plot(data = photo_array)
    #handles.append(handle)
    
    #cbar.ax.set_yscale('log')
    
    ax = plt.gca
    
    # Defining the cursor
    # cursor = Cursor(ax, horizOn=True, vertOn=True, useblit=True,
    #                 color = 'r', linewidth = 1)
    
    
    
    # Add colorbar 
    #cbar = plt.colorbar()
    #cbar.ax.set_yscale('log')
    
    #fig = plt.gcf
    
    #plt.ion
    
    #fig.canvas.mpl_connect('button_press_event', onclick)
    
    #plt.show()


def onclick(event):
    #global coord
    #coord.append((event.xdata, event.ydata))
    x = event.xdata
    y = event.ydata
    z = event.zdata[0]
    
    fig = plt.gcf
    ax = plt.gca
    # Creating an annotating box
    annot = ax.annotate("", xy=(0,0), xytext=(-40,40),textcoords="offset points",
                       bbox=dict(boxstyle='round4', fc='linen',ec='k',lw=1),
                       arrowprops=dict(arrowstyle='-|>'))
    annot.set_visible(False) 
   
    # printing the values of the selected point
    #print([x,y]) 
    annot.xy = (x,y)
    #text = "({:.2g}, {:.2g})".format(x,y)
    annot.set_text(str(x) + '\n' + str(y) + '\n' + str(z))
    annot.set_visible(True)
    fig.canvas.draw() #redraw the figure


    

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
# RUN MAIN PROGRAM
##############################################################################################################
main_program()