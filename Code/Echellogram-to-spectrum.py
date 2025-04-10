# Author: Jasper Ristkok
# v1.5

# Code to convert an echellogram (photo) to spectrum


##############################################################################################################
# CONSTANTS
##############################################################################################################


# '14IWG1A_11a_P11_gate_1000ns_delay_1000ns' r'492_2X8_R7C3C7_\d+? 2X08 - R7 C3-C7' '492_2X8_R7C3C7_0001 2X08 - R7 C3-C7' 'Aryelle'
# 'Plansee_W_3us_gate' 'Plansee_W_' 'IU667_D10_4us' 'IU667_D10' 'integrating_sphere_100ms_10avg_fiber600umB' 
# 'DHLamp_' 'Hg_lamp' 'W_Lamp' 'Ne_lamp_100ms_fiber600umB'
use_sample = 'IU667_D10'
integral_width = 10
first_order_nr = 36

#working_path = 'D:\\Research_analysis\\Projects\\2024_JET\\Lab_comparison_test\\Photo_to_spectrum\\Photos_spectra_comparison\\'
working_path = 'E:\\Nextcloud sync\\Data_processing\\Projects\\2024_09_JET\\Lab_comparison_test\\Photo_to_spectrum\\Photos_spectra_comparison\\'

input_photos_path = working_path + 'Input_photos\\'
input_data_path = working_path + 'Input_data\\'
averaged_path = working_path + 'Averaged\\'
output_path = working_path + 'Output\\'
spectrum_path = working_path + 'Spectra\\'


##############################################################################################################
# IMPORTS
##############################################################################################################

import numpy as np
import math
import os
import re # regex
import json
import tkinter # Tkinter

from PIL import Image as PILImage
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from  matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

#import random
#import scipy.stats
#import scipy.optimize
#from scipy.interpolate import interp2d
#from matplotlib.widgets import Cursor


##############################################################################################################
# DEFINITIONS
# Definitions are mostly structured in the order in which they are called at first.
##############################################################################################################

def main_program():
    
    # create folders if missing
    create_folder_structure()
    
    # Create and draw GUI window
    tkinter_master = tkinter.Tk()
    try:
        window = calibration_window(tkinter_master)
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

def create_folder_structure():
    create_folder(output_path)
    create_folder(averaged_path)
    create_folder(input_photos_path)
    create_folder(input_data_path)
    create_folder(spectrum_path)
    
        

def prepare_photos(use_sample = ''):
    # Get files
    averaged_files = get_folder_files(averaged_path, use_sample = use_sample)
    
    exif_data = None
    
    averaged_name = 'average_' + use_sample + '.tif'
    if averaged_name in averaged_files: # averaging done previously
        average_array, exif_data = load_photo(averaged_path + averaged_name)
    
    # Average all photos in input folder and save averaged photo into averaged folder
    else:
        input_files = get_folder_files(input_photos_path, use_sample = use_sample, identificator = use_sample)
        average_array, exif_data = average_photos(input_files)
        output_averaged_photo(average_array, output_path, use_sample, exif_data)
    
    # Convert spectra to photos for GIMP or something else
    spectra_to_photos()
    
    return average_array, use_sample


def load_photo(filepath):
    image = PILImage.open(filepath)
    #im.show()
    #exif_data = image.info['exif'] # save exif data, so image can be opened with Sophi nXt
    exif_data = image.getexif() # save exif data, so image can be opened with Sophi nXt
    # TODO: get exif data (especially description (or comment) part) and save with new image
    
    imArray = np.array(image)
    return imArray, exif_data


def average_photos(input_files):
    
    # Initialize averaging variables
    count = 0
    sum = None
    
    # iterate over files
    for filename in input_files:
        imArray, exif_data = load_photo(input_photos_path + filename)
        
        # sum arrays
        if sum is None: # initialize
            sum = imArray
        else:  # add
            sum += imArray
        count += 1
    
    average = sum / count
    return average, exif_data


def output_averaged_photo(average_array, output_path, identificator, exif_data = {}):
    #for k, v in exif_data.items():
    #    print("Tag", k, "Value", v)  # Tag 274 Value 2
    
    identificator = identificator.replace(r'_\d+?', '') # strip _0001 in case it was specified
    
    image = PILImage.fromarray(average_array)
    image.save(averaged_path + 'average_' + identificator + '.tif')#, exif = exif_data) 
    
def spectra_to_photos():
    files = get_folder_files(spectrum_path, use_sample = use_sample)
    
    for file in files:
        if file.endswith(".dat") or file.endswith(".csv"):
            #spectrum = np.fromfile(spectrum_path + file, dtype=float, sep=' ') # Todo: more separators
            spectrum = np.loadtxt(spectrum_path + file, delimiter=' ', usecols = (1))
            #load_spectrum(spectrum_path + file)
            
            # Save image
            image = PILImage.fromarray(spectrum)
            #image.show()
            image.save(spectrum_path + file + '.tif') # TODO: strip file extension first


'''
def load_spectrum(filepath):
    array = np.fromfile(filepath, dtype=float, sep=' ', like = np.empty((39303, 2)))
    return
'''


##############################################################################################################
# Processing phase
##############################################################################################################

def process_photo(photo_array, identificator):
    
    # negative values to 0
    photo_array[photo_array < 0] = 0
    
    
    # Load calibration data if it exists
    calibr_data, vertical_calibration_array = load_calibration_data()
    
    #calibration_window(photo_array)
    
    
    np.savetxt( output_path + 'image_array.csv', photo_array, delimiter = ',')
    #draw_pyplot(photo_array, identificator)
    
    return photo_array, calibr_data, vertical_calibration_array

def load_calibration_data():
    calibration_data = None
    
    output_files = get_folder_files(output_path, use_sample = use_sample)
    
    if 'calibration_data.json' in output_files: # calibration done previously
        with open(output_path + 'calibration_data.json', 'r') as file:
            try:
                calibration_data = json.load(file)
            except: # e.g. file is empty
                pass
    
    vertical_calibration_array = None
    input_data_files = get_folder_files(input_data_path, use_sample = use_sample)
    if 'Vertical_points.csv' in input_data_files:
        vertical_calibration_array = np.loadtxt(input_data_path + 'Vertical_points.csv', delimiter = ',', skiprows = 1)
        
            
            
    return calibration_data, vertical_calibration_array

###########################################################################################
# Generic point class
###########################################################################################

# A point on a graph
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

###########################################################################################
# Points classes
###########################################################################################

# Hold calibration data in this format (same as dictionary). x is horizontal and y vertical pixels
class calibration_data():
    
    def __init__(self, image_width):
        self.image_width = image_width
    
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
            point_dict['z'] = point.z
            point_dict['group'] = point.group
            
            points_array.append(point_dict)
            
        save_data = points_array
        return save_data
        
    # Convert saved dictionary into points
    def load_encode(self, existing_data):
        self.points = []
        
        # Iterate over points and encode them
        for point_dict in existing_data:
            self.points.append(point_class(point_dict['x'], y = point_dict['y'], z = point_dict['z'], group = point_dict['group']))
        
        self.update()
        
    
class start_points(calibration_data):
    def __init__(self, image_width = None, existing_data = None):
        super().__init__(image_width)
        
        if existing_data is None:
            self.points = [point_class(0, 0, group = 'start'), 
                        point_class(self.image_width / 4 - 1, self.image_width / 2 - 1, group = 'start'), 
                        point_class(self.image_width / 3 - 1, self.image_width - 1, group = 'start')]
        else:
            self.load_encode(existing_data) 
            
        self.update()
        
class end_points(calibration_data):
    def __init__(self, image_width = None, existing_data = None):
        super().__init__(image_width)
        
        if existing_data is None:
            self.points = [point_class(self.image_width * 2 / 3 - 1, self.image_width - 1, group = 'end'), 
                        point_class(self.image_width * 3 / 4 - 1, self.image_width / 2 - 1, group = 'end'), 
                        point_class(self.image_width - 1, 0, group = 'end')]
        else:
            self.load_encode(existing_data)
        
        self.update()

# Line of a diffraction order (horizontal)
class order_points(calibration_data):
    def __init__(self, image_width = None, existing_data = None):
        super().__init__(image_width)
        self.order_nr = -1
        self.bounds_px = (None, None) # pixel index
        self.bounds_wave = (None, None) # wavelengths
        
        if existing_data is None:
            '''
            self.points = [point_class(0, self.image_width / 2 - 1, group = 'orders'), 
                        point_class(self.image_width / 2 - 1, self.image_width / 2 - 1, group = 'orders'), 
                        point_class(self.image_width - 1, self.image_width / 2 - 1, group = 'orders')]
            '''
            self.points = [point_class(0, 0, group = 'orders'), 
                        point_class(self.image_width / 2 - 1, 0, group = 'orders'), 
                        point_class(self.image_width - 1, 0, group = 'orders')]
            
        else:
            self.load_encode(existing_data)
        
        self.update()

###########################################################################################
# Main class
###########################################################################################

# Window to calibrate the echellogram to spectrum curves
class calibration_window():
    
    # Main code
    def __init__(self, parent): #, photo_array, calibration_dict = None, order_bounds = None):
        
        # Initialize argument variables
        self.root = parent
        #self.photo_array = photo_array
        #self.order_bounds = order_bounds
        
        # Initialize class variables
        self.init_class_variables()
        
        # Draw main window and tkinter elements
        self.initialize_window_elements()
        
        self.load_sample_data()
        
        
        # Load input data
        #self.load_data(calibration_dict)
        
        # Draw matplotlib plot
        self.initialize_plot()
        
        # Draw tkinter window elements
        self.pack_window_elements()
        
    
    #######################################################################################
    # Initialize class
    #######################################################################################
    
    def init_class_variables(self):
        self.use_sample = use_sample
        self.working_path = working_path
        self.first_order_nr = first_order_nr
        self.integral_width = integral_width
        
        self.show_verticals = False
        self.show_orders = True
        self.autoupdate_spectrum = True
        self.program_mode = None
        self.selected_order = None
        self.best_order_idx = None
        self.autoscale_spectrum = True
        
        
        self.photo_array = []
        self.order_bounds = []
        self.order_plot_curves = []
        self.order_plot_points = []
        self.order_poly_coefs = []
        self.order_bound_points = []
        
        # dictionary to store three calibration things (start and end vertical, nr of horizontal)
        self.calib_data = None
    
    
    #######################################################################################
    # Tkinter window elements
    #######################################################################################
    
    # Create tkinter window elements
    def initialize_window_elements(self):
        #self.root.winfo_width()
        #self.root.winfo_height()
        #self.root.winfo_screenwidth()
        #self.root.winfo_screenheight()
        
        op_sys_toolbar_height = 0.07 #75 px for 1080p
        self.root.maxsize(self.root.winfo_screenwidth(), self.root.winfo_screenheight() - round(self.root.winfo_screenheight() * op_sys_toolbar_height))
        
        self.frame_spectrum = tkinter.Frame(self.root)#, height = self.root.winfo_height() / 3)
        self.frame_buttons_image = tkinter.Frame(self.root)
        self.frame_image = tkinter.Frame(self.frame_buttons_image)
        self.frame_buttons_input = tkinter.Frame(self.frame_buttons_image)
        self.frame_input = tkinter.Frame(self.frame_buttons_input)
        self.frame_buttons = tkinter.Frame(self.frame_buttons_input)
        
        # Inputs
        self.use_sample_var = tkinter.StringVar()
        self.input_path_var = tkinter.StringVar()
        self.input_order_var = tkinter.StringVar()
        self.integral_width_var = tkinter.StringVar()
        self.shift_orders_var = tkinter.StringVar()
        input_sample_label = tkinter.Label(self.frame_input, text = 'Use sample:')
        input_sample = tkinter.Entry(self.frame_input, textvariable = self.use_sample_var)
        input_path_label = tkinter.Label(self.frame_input, text = 'Directory path:')
        input_path = tkinter.Entry(self.frame_input, textvariable = self.input_path_var)
        input_order_label = tkinter.Label(self.frame_input, text = 'First order nr:')
        input_order = tkinter.Entry(self.frame_input, textvariable = self.input_order_var)
        input_width_label = tkinter.Label(self.frame_input, text = 'Integral width:')
        input_width = tkinter.Entry(self.frame_input, textvariable = self.integral_width_var)
        input_shift_label = tkinter.Label(self.frame_input, text = 'Shift orders up:')
        input_shift_orders = tkinter.Entry(self.frame_input, textvariable = self.shift_orders_var)
        save_variables_btn = tkinter.Button(self.frame_input, text = "Save variables", command = self.save_variables)
        
        self.use_sample_var.set(self.use_sample)
        self.input_path_var.set(str(self.working_path))
        self.input_order_var.set(str(self.first_order_nr))
        self.integral_width_var.set(str(self.integral_width))
        self.shift_orders_var.set(0)
        
        
        input_sample_label.grid(row = 0, column = 0)
        input_sample.grid(row = 0, column = 1)
        input_path_label.grid(row = 1, column = 0)
        input_path.grid(row = 1, column = 1)
        input_order_label.grid(row = 2, column = 0)
        input_order.grid(row = 2, column = 1)
        input_width_label.grid(row = 3, column = 0)
        input_width.grid(row = 3, column = 1)
        input_shift_label.grid(row = 4, column = 0)
        input_shift_orders.grid(row = 4, column = 1)
        save_variables_btn.grid(row = 5, column = 0)
        
        
        # Labels
        self.feedback_label = tkinter.Label(self.frame_buttons, text = '')
        self.feedback_label.pack(side = 'top')
        
        
        # Create buttons
        self.load_order_points_btn = tkinter.Button(self.frame_buttons, text = "Load order points", command = self.load_order_points)
        self.button_save_coefs_btn = tkinter.Button(self.frame_buttons, text = "Output stuff", command = self.write_outputs)
        self.button_save_btn = tkinter.Button(self.frame_buttons, text = "Save calibr points", command = self.save_data)
        self.button_reset_btn = tkinter.Button(self.frame_buttons, text = "Reset mode", command = self.reset_mode)
        self.vertical_mode_btn = tkinter.Button(self.frame_buttons, text = "Vertical curve edit mode", command = self.bounds_mode)
        self.orders_mode_btn = tkinter.Button(self.frame_buttons, text = "Diffr order curve edit mode", command = self.orders_mode)
        self.orders_tidy_btn = tkinter.Button(self.frame_buttons, text = "Tidy order points", command = self.orders_tidy)
        
        
        add_order_btn = tkinter.Button(self.frame_buttons, text = "Add diffr order", command = self.add_order)
        delete_order_btn = tkinter.Button(self.frame_buttons, text = "Delete diffr order", command = self.delete_order)
        
        self.toggle_spectrum_update_btn = tkinter.Button(self.frame_buttons, text = "Toggle spectrum updating", command = self.toggle_spectrum_update)
        self.toggle_show_orders_btn = tkinter.Button(self.frame_buttons, text = "Toggle showing orders", command = self.toggle_show_orders)
        #self.add_point_left = tkinter.Button(self.frame_buttons, text = "Add point to left vertical", command = self.add_calibr_point_left)
        #self.add_point_right = tkinter.Button(self.frame_buttons, text = "Add point to right vertical", command = self.add_calibr_point_right)
        #self.add_point_order = tkinter.Button(self.frame_buttons, text = "Add point to order", command = self.add_calibr_point_order)
        self.spectrum_log_btn = tkinter.Button(self.frame_buttons, text = "Toggle spectrum log scale", command = self.spectrum_toggle_log)
        update_scale_btn = tkinter.Button(self.frame_buttons, text = "Toggle autoupdate spectrum scale", command = self.update_scale_fn)
        
        
        self.image_int_down_btn = tkinter.Button(self.frame_buttons, text = "Lower colorbar max x2", command = self.image_int_down)
        self.image_int_up_btn = tkinter.Button(self.frame_buttons, text = "Raise colorbar max x5", command = self.image_int_up)
        self.image_int_min_down_btn = tkinter.Button(self.frame_buttons, text = "Lower colorbar min x2", command = self.image_int_min_down)
        self.image_int_min_up_btn = tkinter.Button(self.frame_buttons, text = "Raise colorbar min x5", command = self.image_int_min_up)
        
        
        
        # Show buttons
        self.load_order_points_btn.pack()
        self.button_save_coefs_btn.pack()
        self.button_save_btn.pack()
        self.toggle_spectrum_update_btn.pack()
        self.button_reset_btn.pack()
        self.vertical_mode_btn.pack()
        self.orders_mode_btn.pack()
        self.orders_tidy_btn.pack()
        add_order_btn.pack()
        delete_order_btn.pack()
        
        self.spectrum_log_btn.pack()
        self.toggle_show_orders_btn.pack()
        update_scale_btn.pack()
        
        self.image_int_down_btn.pack()
        self.image_int_up_btn.pack()
        self.image_int_min_down_btn.pack()
        self.image_int_min_up_btn.pack()
        
        self.root.title("Echellogram calibration")
        #self.root.configure(background="yellow")
        self.root.minsize(600, 600)
        #self.root.maxsize(500, 500)
        #self.root.geometry("300x300+50+50")
        
        #self.root.update()
        #self.pack_window_elements()
        
    # Draw window elements
    def pack_window_elements(self):
        self.frame_buttons_image.pack(fill='both', expand=1, side = 'top')
        self.frame_spectrum.pack(fill='both', expand=1, side = 'top')
        self.frame_buttons_input.pack(side = 'left')
        self.frame_image.pack(fill='both', expand=1, side = 'left')
        self.frame_input.pack(side = 'top')
        self.frame_buttons.pack(side = 'top')
        
        
    #######################################################################################
    # Load sample data
    #######################################################################################
    
    def load_sample_data(self):
        
        # average input photos and return the array
        average_array, identificator = prepare_photos(self.use_sample)
        
        # assumes square image
        if (identificator != self.use_sample):
            print('Sample not found in input files')
            self.set_feedback('Sample not found in input files', 10000)
            
        photo_array, calibration_data, vertical_calibration_array = process_photo(average_array, identificator)
        
        self.photo_array = photo_array
        self.order_bounds = vertical_calibration_array
        
        # Load input data
        if self.calib_data is None: # only initialize diffr orders once after program start
            self.calib_data = {}
            self.load_data(calibration_data)
        
    
    def load_data(self, calibration_dict):
        
        # initialize calib_data with photo_array bounds for better plotting
        if calibration_dict is None:
            self.calib_data['start'] = start_points(image_width = self.photo_array.shape[0])
            self.calib_data['end'] = end_points(image_width = self.photo_array.shape[0])
            self.calib_data['orders'] = [order_points(image_width = self.photo_array.shape[0])]
            
            
        else: # load saved data
            self.calib_data['start'] = start_points(existing_data = calibration_dict['start'])
            self.calib_data['end'] = end_points(existing_data = calibration_dict['end'])
            
            # Iterate over orders
            self.calib_data['orders'] = []
            for order_raw in calibration_dict['orders']:
                self.calib_data['orders'].append(order_points(existing_data = order_raw))
                
                
            # sort orders by average y values
            self.sort_orders(sort_plots = False)
    
    
    #######################################################################################
    # Button functions
    #######################################################################################
    
    def save_variables(self):
        new_sample = self.use_sample_var.get()
        self.working_path = self.input_path_var.get()
        self.first_order_nr = int(self.input_order_var.get())
        self.integral_width = float(self.integral_width_var.get())
        shift_orders_amount = float(self.shift_orders_var.get())
        
        global input_photos_path, input_data_path, averaged_path, output_path, spectrum_path
        input_photos_path = self.working_path + 'Input_photos\\'
        input_data_path = self.working_path + 'Input_data\\'
        averaged_path = self.working_path + 'Averaged\\'
        output_path = self.working_path + 'Output\\'
        spectrum_path = self.working_path + 'Spectra\\'
        
        
        
        self.shift_orders(shift_orders_amount)
        
        # Empty the variables
        #self.input_path_var.set('')
        #self.input_order_var.set('')
        #self.integral_width_var.set('')
        self.shift_orders_var.set(0)
        
        # Use new sample, load data again and draw plots again
        if new_sample != self.use_sample:
            
            # Load new sample data
            self.use_sample = new_sample
            self.load_sample_data()
            
            # Draw plots again
            self.initialize_plot(reset = True)
            
        
        self.update_all()
        
        self.set_feedback('Input variables saved')
    
    
    # Read in csv file and overwrite order points based on these
    def load_order_points(self):
        
        all_points = np.loadtxt(input_data_path + 'Order_points.csv', delimiter=',', skiprows = 1)
        
        for idx in range(len(self.calib_data['orders'])):
            order = self.calib_data['orders'][idx]
            
            # get order_nr row index from data
            order_nr = order.order_nr
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
       self.set_feedback('Output written')
   
    # Save the polynomial coefficients of diffraction orders into output
    def output_coefs(self):
        
        with open(output_path + 'order_coefficients.csv', 'w') as file:
            file.write('Order_nr,Coef1,Coef2,Coef3\n') # headers
            
            for idx in range(len(self.calib_data['orders'])):
                order = self.calib_data['orders'][idx]
                coefs = self.order_poly_coefs[idx]
                file.write(str(order.order_nr))
                
                for coef in coefs:
                    file.write(',' + str(coef))
                
                file.write('\n')
        
        
    
    def output_order_points(self):
        with open(output_path + 'order_points.csv', 'w') as file:
            file.write('Order_nr,Point0_x,Point0_y,Point1_x,Point1_y,Point2_x,Point2_y\n') # headers
            
            for idx in range(len(self.calib_data['orders'])):
                order = self.calib_data['orders'][idx]
                file.write(str(order.order_nr))
                
                for point in order.points:
                    file.write(',' + str(point.x))
                    file.write(',' + str(point.y))
                
                file.write('\n')
        
    
    # Save calibration curves and the corresponding points
    def save_data(self):
        
        # decode objects into dictionaries
        save_dict = {}
        save_dict['start'] = self.calib_data['start'].save_decode()
        save_dict['end'] = self.calib_data['end'].save_decode()
        save_dict['orders'] = [] 
        
        # Iterate over orders
        for order in self.calib_data['orders']:
            save_dict['orders'].append(order.save_decode())
        
        # Save as JSON readable output
        with open(output_path + 'calibration_data.json', 'w') as file:
            json.dump(save_dict, file, sort_keys=True, indent=4)
            
        self.set_feedback('Calibration data saved')
    
    
    def reset_mode(self):
        self.program_mode = None
        self.selected_order = None
        
        self.show_verticals = False
        self.hide_unhide_verticals()
        
        self.set_feedback('Mode: reset', 1000)
        
        
    def bounds_mode(self):
        self.program_mode = 'bounds'
        self.selected_order = None
        
        self.show_verticals = True
        self.hide_unhide_verticals()
        
        self.set_feedback('Mode: vertical curves', 1000)
        
        
    def orders_mode(self):
        self.program_mode = 'orders'
        self.selected_order = None
        
        self.show_orders = True
        #self.update_order_curves()
        self.hide_unhide_orders()
        
        self.set_feedback('Mode: horizontal curves', 1000)
    
    # Overwrite the order points with the ones calculated from polynomials
    def orders_tidy(self):
        start_x = 20
        center_x = math.floor(self.photo_array.shape[0] / 2)
        end_x = self.photo_array.shape[0] - 20
        
        for idx in range(len(self.calib_data['orders'])):
            order = self.calib_data['orders'][idx]
            
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
        self.calib_data['orders'] += [diffr_order]
        self.plot_order(diffr_order)
        
        # sort orders by average y values
        #self.calib_data['orders'].sort(key=lambda x: x.avg_y, reverse=True)
        
        self.selected_order = diffr_order
        self.best_order_idx = len(self.calib_data['orders']) - 1
        
        # Calculate the bounds for that order
        self.initialize_bounds(self.best_order_idx)
        self.draw_bounds(self.best_order_idx)
        
        self.sort_orders()
        
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
        
        
        
        #self.update_order_curves()
        self.update_spectrum(autoscale = True)
        
        
        # Draw plot again and wait for drawing to finish
        self.canvas_spectrum.draw()
        self.canvas_spectrum.flush_events()
        
        self.set_feedback('Order added and selected', 1000)
    
    def delete_order(self):
        
        if self.best_order_idx is None:
            self.set_feedback('No order selected, use diff edit mode first', 8000)
            return
        
        # Show orders
        self.show_orders = True
        self.hide_unhide_orders()
        
        selected_idx = self.best_order_idx
        selected_order = self.calib_data['orders'][selected_idx]
        order_nr = selected_order.order_nr
        
        del self.calib_data['orders'][selected_idx]
        
        # Deselect the order
        self.selected_order = None
        self.best_order_idx = None
        
        # Draw plots again
        self.initialize_plot(reset = True)
        self.update_all()
        
        #self.update_spectrum(autoscale = True)
        
        self.reset_mode()
        
        self.set_feedback('Order deleted: ' + order_nr, 5000)
    
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
        
        
        # Modify vertical curves
        if self.program_mode == 'bounds':
            all_points = gather_points([self.calib_data['start'].points, self.calib_data['end'].points])
            
            # get closest calibration point
            min_distance = math.inf
            #best_point = None
            #best_point_idx = 0
            for idx in range(len(all_points)):
                distance = all_points[idx].distance(click_point)
                if distance < min_distance:
                    best_point_idx = idx
                    min_distance = distance
                    best_point = all_points[idx]
            
            #print('best: ', best_point.group, best_point_idx, best_point.x, best_point.y)
            
            # convert idx to index of first or second list
            group_idx = best_point_idx
            if best_point_idx > (len(self.calib_data['start'].points) - 1):
                group_idx = best_point_idx - len(self.calib_data['start'].points)
            
            
            click_point.group = best_point.group
            self.calib_data[best_point.group].points[group_idx] = click_point
            
            self.calib_data['start'].update()
            self.calib_data['end'].update()
            self.update_vertical_curves()
            
        
        # Modify horizontal curves
        elif self.program_mode == 'orders':
            
            if not self.show_orders:
                return
            
            # Iterate over diffraction orders, select the point closest to the click
            min_distance = math.inf
            if self.selected_order is None:
                for idx in range(len(self.calib_data['orders'])):
                    order = self.calib_data['orders'][idx]
                    
                    # iterate over points in order, save best point
                    for idx2 in range(len(order.points)):
                        distance = order.points[idx2].distance(click_point)
                        if distance < min_distance:
                            best_order_idx = idx
                            min_distance = distance
                
                self.selected_order = self.calib_data['orders'][best_order_idx]
                self.best_order_idx = best_order_idx # TODO: delete hack
                self.set_feedback('Order ' + str(best_order_idx + self.first_order_nr) + 'selected')
                return
            
            # Iterate over points of the selected orded
            for idx in range(len(self.selected_order.points)):
                distance = self.selected_order.points[idx].distance(click_point)
                if distance < min_distance:
                    best_point_idx = idx
                    min_distance = distance
                    best_point = self.selected_order.points[idx]
            
            #print('best: ', best_point.group, best_point_idx, best_point.x, best_point.y)
            
            # edit point
            click_point.group = best_point.group
            self.calib_data['orders'][self.best_order_idx].points[best_point_idx] = click_point # TODO: delete hack 
            self.calib_data['orders'][self.best_order_idx].update()
            
            # Calculate the bounds for that order
            self.initialize_bounds(self.best_order_idx)
            
            # sort orders by average y values
            self.sort_orders()
            
            self.update_order_curves()
            
        
        if (not self.autoupdate_spectrum) or (self.program_mode is None):
            return
            
        # Redraw plot
        self.update_spectrum()
    
    
    def shift_orders(self, shift_amount = 0):
        if shift_amount == 0:
            return
        
        for idx in range(len(self.calib_data['orders'])):
            order = self.calib_data['orders'][idx]
            
            for point in order.points:
                point.y -= shift_amount # shift up, so coordinate decreases
        
        
        self.set_feedback('Orders shifted up by ' + str(shift_amount) + ' px')
        
    
    #######################################################################################
    # Initialize Matplotlib plot
    #######################################################################################
    
    def initialize_plot(self, reset = False):
        
        if reset:
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
        self.plot_verticals()
        self.plot_orders()
        self.draw_all_bounds()
    
    def plot_verticals(self):
        # Get line data
        y_values = np.arange(self.photo_array.shape[0])
        left_curve_array, self.left_poly_coefs = get_polynomial_points(self.calib_data['start'], self.photo_array.shape[0], flip = True)
        right_curve_array, self.right_poly_coefs = get_polynomial_points(self.calib_data['end'], self.photo_array.shape[0], flip = True)
        
        # Plot curves
        self.start_curve, = self.plot_ax.plot(left_curve_array, y_values, 'tab:orange')
        self.end_curve, = self.plot_ax.plot(right_curve_array, y_values, 'tab:orange')
        
        # Plot calibration points
        self.start_curve_points, = self.plot_ax.plot(self.calib_data['start'].xlist, self.calib_data['start'].ylist, color = 'k', marker = 'o', linestyle = '', markersize = 4)
        self.end_curve_points, = self.plot_ax.plot(self.calib_data['end'].xlist, self.calib_data['end'].ylist, color = 'k', marker = 'o', linestyle = '', markersize = 4)
        
        # Hide if needed
        self.hide_unhide_verticals()
    
    # Iterate over diffraction orders and plot them
    def plot_orders(self):
        x_values = np.arange(self.photo_array.shape[0])
        for idx in range(len(self.calib_data['orders'])):
            order = self.calib_data['orders'][idx]
            self.plot_order(order, x_values)
        
    
    def plot_order(self, order, x_values = None):
        if x_values is None:
            x_values = np.arange(self.photo_array.shape[0])
        
        
        # Get line data
        curve_array, poly_coefs = get_polynomial_points(order, self.photo_array.shape[0])
        self.order_poly_coefs.append(poly_coefs)
        
        # Plot curve
        curve, = self.plot_ax.plot(x_values, curve_array, 'r')
        self.order_plot_curves.append(curve)
        
        # Plot calibration points
        curve_points, = self.plot_ax.plot(order.xlist, order.ylist, color = mcolors.CSS4_COLORS['peru'], marker = 'o', linestyle = '', markersize = 4)
        self.order_plot_points.append(curve_points)
        
    # Draw points for the bounds even if they aren't calculated correctly yet
    def draw_all_bounds(self):
        for order_idx in range(len(self.calib_data['orders'])):
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
        x_values, z_values = self.get_order_spectrum()
        self.spectrum_curve, = self.spectrum_ax.plot(x_values, z_values, 'tab:pink')
        
        # rescale to fit
        self.spectrum_ax.set_xlim(min(x_values), max(x_values))
        self.spectrum_ax.set_ylim(min(z_values), max(z_values))
        
    
    #######################################################################################
    # Update visual things
    #######################################################################################
    
    def clear_feedback(self):
        self.feedback_label.config(text = '')
    
    def set_feedback(self, string, delay = 3000):
        self.feedback_label.config(text = string)
        self.feedback_label.after(delay, self.clear_feedback) # clear feedback after delay
    
    
    def update_all(self):
        
        # Deselect order
        self.best_order_idx = None
        self.set_feedback('Order deselected')
        
        self.update_point_instances()
        
        self.update_vertical_curves()
        self.update_order_curves()
        
        # sort orders by average y values
        self.sort_orders()
        
        self.calculate_all_bounds(initialize_x = True)
            
        
        self.update_spectrum(autoscale = True)
        
        # Draw things again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events()
        self.canvas_spectrum.draw()
        self.canvas_spectrum.flush_events()
    
    # Update instances of point classes
    def update_point_instances(self):
        self.calib_data['start'].update()
        self.calib_data['end'].update()
        for idx in range(len(self.calib_data['orders'])):
            self.calib_data['orders'][idx].update()
    
    # Redraw vertical calibration curves
    def update_vertical_curves(self):
        
        if not self.show_verticals:
            return
        
        # points
        self.start_curve_points.set_xdata(self.calib_data['start'].xlist)
        self.start_curve_points.set_ydata(self.calib_data['start'].ylist)
        self.end_curve_points.set_xdata(self.calib_data['end'].xlist)
        self.end_curve_points.set_ydata(self.calib_data['end'].ylist)
        
        
        # Get line data
        left_curve_array, self.left_poly_coefs = get_polynomial_points(self.calib_data['start'], self.photo_array.shape[0], flip = True)
        right_curve_array, self.right_poly_coefs = get_polynomial_points(self.calib_data['end'], self.photo_array.shape[0], flip = True)
        
        # Plot curves
        self.start_curve.set_xdata(left_curve_array)
        self.end_curve.set_xdata(right_curve_array)
        
        self.update_order_curves()
        
        self.calculate_all_bounds(initialize_x = True, use_verticals = True)
        
        self.update_all_bounds(use_verticals = True) # TODO: fix hack (bounds need previous order update)
        
        self.set_feedback('Bounds overwritten with polynomial data', 5000)
        
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
    
    
    # Redraw horizontal calibration curves
    def update_order_curves(self):
        
        if (not self.show_orders) and (self.program_mode != 'bounds'):
            return
        
        # Iterate over diffraction orders
        for idx in range(len(self.calib_data['orders'])):
            order = self.calib_data['orders'][idx]
            
            # curve
            curve_array, poly_coefs = get_polynomial_points(order, self.photo_array.shape[0])
            self.order_poly_coefs[idx] = poly_coefs
            self.order_plot_curves[idx].set_ydata(curve_array)
            
            # Calculate the new bounds for that order
            self.initialize_bounds(idx)
            
            # points
            self.order_plot_points[idx].set_xdata(order.xlist)
            self.order_plot_points[idx].set_ydata(order.ylist)
        
        
        self.update_all_bounds()
            
            
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
    
    # TODO: vertical curve edit mode doesn't change bounds on graph
    def update_all_bounds(self, use_verticals = False):
        for idx in range(len(self.calib_data['orders'])):
            self.update_one_bounds(idx, use_verticals = use_verticals)
            
    # Re-draw bounds
    def update_one_bounds(self, order_idx, use_verticals = False):
        x_start, x_end, y_start, y_end = self.calculate_bounds(order_idx, initialize_x = True, use_verticals = use_verticals)
        self.order_bound_points[order_idx].set_xdata([x_start, x_end])
        self.order_bound_points[order_idx].set_ydata([y_start, y_end])
        
    
    def update_spectrum(self, autoscale = False):
         
         x_values, z_values = self.get_order_spectrum()
         
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
     
    def hide_unhide_verticals(self):
        self.start_curve.set_visible(self.show_verticals)
        self.start_curve_points.set_visible(self.show_verticals)
        self.end_curve.set_visible(self.show_verticals)
        self.end_curve_points.set_visible(self.show_verticals)
        
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
    
    def hide_unhide_orders(self):
        
        # Iterate over diffraction orders
        for idx in range(len(self.calib_data['orders'])):
            #order = self.calib_data['orders'][idx]
            
            self.order_plot_curves[idx].set_visible(self.show_orders)
            self.order_plot_points[idx].set_visible(self.show_orders)
            self.order_bound_points[idx].set_visible(self.show_orders)
        
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
            
    
    
    #######################################################################################
    # Misc
    #######################################################################################
    
    def calculate_all_bounds(self, initialize_x = False, use_verticals = False):
        for idx in range(len(self.calib_data['orders'])):
            self.calculate_bounds(idx, initialize_x = initialize_x, use_verticals = use_verticals)
    
    def calculate_bounds(self, order_idx, initialize_x = False, use_verticals = False):
        order = self.calib_data['orders'][order_idx]
        (px_start, px_end) = order.bounds_px
        
        if initialize_x or use_verticals or (px_start is None) or (px_end is None):
            px_start, px_end = self.initialize_bounds(order_idx, use_verticals = use_verticals)
        
        coefs = np.polynomial.polynomial.polyfit(order.xlist, order.ylist, 2)
        y_start = poly_func_value(px_start, coefs)
        y_end = poly_func_value(px_end, coefs)
        
        return px_start, px_end, y_start, y_end
    
    # Calculate bounds by the crossing of curves and save into order points class
    def initialize_bounds(self, order_idx, use_verticals = False):
        nr_pixels = self.photo_array.shape[0]
        
        curve_x = self.order_plot_curves[order_idx].get_xdata()
        curve_y = self.order_plot_curves[order_idx].get_ydata()
        
        # sort the arrays in increasing x value
        curve_x, curve_y = sort_related_arrays(curve_x, curve_y) 
        
        
        # Calculate by vertical polynomials
        if use_verticals or (self.order_bounds is None):
            left_curve = self.start_curve.get_xdata()
            right_curve = self.end_curve.get_xdata()
            px_start, px_end = get_order_bounds(nr_pixels, curve_x, curve_y, left_curve, right_curve)
        
        # By default calculate by input data
        else:
            
            # get order_nr row index from data
            order_nr = self.calib_data['orders'][order_idx].order_nr
            row_idxs = np.where(self.order_bounds[:,0] == order_nr)[0] # get indices of rows where order nr is the same
            
            # Get bounds
            if len(row_idxs) > 0: # There's match
                row_idx = row_idxs[0]
                px_start = self.order_bounds[row_idx, 1]
                px_end = self.order_bounds[row_idx, 2]
                
                # File contains info about wavelengths
                if self.order_bounds.shape[1] > 3:
                    wave_start = self.order_bounds[row_idx, 3]
                    wave_end = self.order_bounds[row_idx, 4]
                    self.calib_data['orders'][order_idx].bounds_wave = (wave_start, wave_end)
                
            else: # no match, use photo bounds
                px_start = 0
                px_end = nr_pixels
        
        self.calib_data['orders'][order_idx].bounds_px = (px_start, px_end)
        
        
        return px_start, px_end
    
        
    def get_idx_by_order_nr(self, order_nr):
        for idx in range(len(self.calib_data['orders'])):
            order = self.calib_data['orders'][idx]
            if order.order_nr == order_nr:
                return idx
        return None
    
    
    
    # Assumes that each order instance has been updated (avg_y is correct)
    def sort_orders(self, sort_plots = True):
        
        # Get indices to sort by
        y_values = [obj.avg_y for obj in self.calib_data['orders']]
        sorted_idx = np.argsort(y_values) # top curves are low order nrs
        
        # Sort orders
        #self.calib_data['orders'].sort(key=lambda x: x.avg_y, reverse=True)
        self.calib_data['orders'] = [self.calib_data['orders'][idx] for idx in sorted_idx]
        
        # Sort corresponding plot objects if they're initialized
        if sort_plots:
            self.order_plot_points = [self.order_plot_points[idx] for idx in sorted_idx]
            self.order_poly_coefs = [self.order_poly_coefs[idx] for idx in sorted_idx]
            self.order_plot_curves = [self.order_plot_curves[idx] for idx in sorted_idx]
            self.order_bound_points = [self.order_bound_points[idx] for idx in sorted_idx]
        
        # Save order number in objects
        for idx in range(len(self.calib_data['orders'])):
            order = self.calib_data['orders'][idx].order_nr = idx + self.first_order_nr
            
        # Re-select the correct order
        idxs = np.where(sorted_idx == self.best_order_idx)[0] # get indices of rows where idx is the same
        self.best_order_idx = None if len(idxs) == 0 else idxs[0]
        
    
    
    
    
   
    # Get z-values of corresponding x and y values and cut them according to left and right curves
    def get_order_spectrum(self):
        x_values = None
        z_values = []
        
        
        '''
        # load left-right calibration curve data
        x2 = self.start_curve.get_xdata()
        y2 = self.start_curve.get_ydata().astype(np.float64)
        x3 = self.end_curve.get_xdata()
        y3 = self.end_curve.get_ydata()
        
        # sort the arrays in increasing x value
        x2, y2 = sort_related_arrays(x2, y2)
        x3, y3 = sort_related_arrays(x3, y3)
        '''
        nr_pixels = self.photo_array.shape[0]
        photo_pixels = np.arange(nr_pixels)
        
        # Compile z-data from the photo with all diffraction orders
        # Iterate over orders backwards because they are sorted by order nr (top to bottom)
        for order_idx in range(len(self.calib_data['orders']) - 1, -1, -1):
            #order = self.calib_data['orders'][order_idx]
            
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
            
            #left_curve = self.start_curve.get_xdata()
            #right_curve = self.end_curve.get_xdata()
            
            # Get bounds for this diffraction order
            #x_left, x_right = get_order_bounds(nr_pixels, curve_x, curve_y, left_curve, right_curve)
            #x_left = get_polynomial_intersection(self.order_poly_coefs[order_idx], self.left_poly_coefs, nr_pixels, is_left = True)
            #x_right = get_polynomial_intersection(self.order_poly_coefs[order_idx], self.right_poly_coefs, nr_pixels, is_left = False)
            
            #print(x_left, x_right)
            (x_left, x_right) = self.calib_data['orders'][order_idx].bounds_px
            
            # Default bounds as first px and last px
            if x_left is None:
                x_left = 0
            if x_right is None:
                x_right = nr_pixels - 1
            
            # get z-values corresponding to the interpolated pixels on the order curve
            for idx2 in photo_pixels: # TODO: add width integration
                
                # Save px only if it's between left and right calibration curves
                x_px = curve_x[idx2]
                
                # Do stuff if point is between bounds
                #if self.point_in_range(x_value, curve_x, curve_y, x2, y2, x3, y3): # intersection function is way too slow
                if x_left <= x_px <= x_right:
                    y_px = y_pixels[idx2]
                    
                    (wave_start, wave_end) = self.calib_data['orders'][order_idx].bounds_wave
                    
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
                        
                    
                    # Get z value from Echellogram
                    integral = integrate_order_width(self.photo_array, x_px, y_px, width = self.integral_width)
                    z_values.append(integral) # x and y have to be switched (somewhy)
                
        return x_values, z_values
    
    

    def get_wavelength(self, order_idx, x_px):
        
        # Get bounds
        (px_start, px_end) = self.calib_data['orders'][order_idx].bounds_px
        (wave_start, wave_end) = self.calib_data['orders'][order_idx].bounds_wave
        
        # Do linear regression and find the wavelength of x_px
        d_px = px_end - px_start
        d_wave = wave_end - wave_start
        slope = d_wave / d_px
        intercept = wave_start - slope * px_start
        
        wavelength = x_px * slope + intercept
        return wavelength

    
    def point_in_range(self, x_point, x1, y1, x2, y2, x3, y3):
        
        x_start,_ = intersection(x1,y1,x2,y2) # intersection function is way too slow
        x_start = x_start[0]
        
        x_end,_ = intersection(x1,y1,x3,y3)
        x_end = x_end[0]
        
        if x_start <= x_point <= x_end:
            return True
        return False
    


# Sum pixels around the order
# If the width is even number then take asymmetrically one pixel from lower index
# if use_weights is True then the integral is summed with Gaussian weights (max is in center) with FWHM of width
def integrate_order_width(photo_array, x_pixel, y_pixel, width = 1, use_weights = True):
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


# Get intersections between order and left-right vertical curves. Assumes non-crazy verticals (not over half)
def get_order_bounds(image_size, order_curve_x, order_curve_y, left_curve, right_curve):
    
    # Initialize
    x_start = 0
    x_end = image_size - 1
    
    # TODO: more efficient algorithm or figure out polynomial rotation in a foolproof way
    
    sweet_spot_distance = 3
    
    # Iterate over pixels towards left, starting from center
    for idx in range(math.ceil(image_size / 2) - 1, -1, -1):
        order_x = idx
        order_y = order_curve_y[idx]
        vertical_y = round(order_y)
        vertical_x = left_curve[vertical_y] # self.start_curve has x-values in ascending order of y
        
        # save point if closer to intersection than previous
        min_dist = math.inf
        close_to_intersection = False
        if (abs(order_x - vertical_x)  < sweet_spot_distance) and (abs(order_y - vertical_y) < sweet_spot_distance): # within 3 pixels of intersection
            close_to_intersection = True
            order_point = point_class(order_x, order_y)
            vertical_point = point_class(vertical_x, vertical_y)
            
            distance = order_point.distance(vertical_point)
            if distance < min_dist:
                min_dist = distance
                x_start = order_point.x
        
        elif close_to_intersection and (abs(order_x - vertical_x)  > sweet_spot_distance) and (abs(order_y - vertical_y) > sweet_spot_distance): # past the sweet spot, save calculation time
            break
    
    # Iterate over pixels towards right, starting from center
    for idx in range(math.ceil(image_size / 2) - 1, image_size):
        order_x = idx
        order_y = order_curve_y[idx]
        vertical_y = round(order_y)
        vertical_x = right_curve[vertical_y] # self.start_curve has x-values in ascending order of y
        
        # save point if closer to intersection than previous
        min_dist = math.inf
        close_to_intersection = False
        if (abs(order_x - vertical_x)  < sweet_spot_distance) and (abs(order_y - vertical_y) < sweet_spot_distance): # within 3 pixels of intersection
            close_to_intersection = True
            order_point = point_class(order_x, order_y)
            vertical_point = point_class(vertical_x, vertical_y)
            
            distance = order_point.distance(vertical_point)
            if distance < min_dist:
                min_dist = distance
                x_end = order_point.x
        
        elif close_to_intersection and (abs(order_x - vertical_x)  > sweet_spot_distance) and (abs(order_y - vertical_y) > sweet_spot_distance): # past the sweet spot, save calculation time
            break
    
    
    return x_start, x_end



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
def get_polynomial_points(calib_data, arr_length, flip = False):
    # get regression coeficients
    poly_coefs = np.polynomial.polynomial.polyfit(calib_data.xlist, calib_data.ylist, 2)
    
    if flip: # treat x as y and y as x
        use_coefs = np.polynomial.polynomial.polyfit(calib_data.ylist, calib_data.xlist, 2)
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
def get_folder_files(path, use_sample = '', identificator = None):
    # Get files, not directories
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
    # Extract files with the identificator
    if identificator is None:
        return onlyfiles
    else:
        '''
        # Get output file identificator
        name = input_files[0] # identificator_0025.tif
        pattern = r'^(.+?)_\d+?\.tif$' # (any characters), _, digits, ., tif
        identificator = re.search(pattern, name)
        if identificator: # match found, extract value
            identificator = identificator.group(1)
        else: # no match
            identificator = ''
        '''
        
        
        pattern = '^' + use_sample 
        if not r'_\d+?' in pattern: # add _0001.tif to the end of pattern unless already specified
            pattern += r'_\d+?' # sample name, _, digits, ., tif
        if not r'\.tif$' in pattern:
            pattern += '\.tif$'
        
        
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
    for idx in range(len(coefs)):
        value += coefs[idx] * x_value ** idx
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
# RUN MAIN PROGRAM
##############################################################################################################
main_program()