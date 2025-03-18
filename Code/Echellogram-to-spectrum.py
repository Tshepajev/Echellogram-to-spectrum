# Author: Jasper Ristkok
# v1.0

# Code to convert an echellogram (photo) to spectrum



use_sample = 'DHLamp_' #'Hg_lamp' #'W_Lamp'
integral_width = 5

##############################################################################################################
# IMPORTS
##############################################################################################################
from matplotlib import pyplot as plt
#import matplotlib.colors as mcolors
#import scipy.stats
import math
#import scipy.optimize
import numpy as np
import os
#import random

import json

import re # regex

from PIL import Image as PILImage

from scipy.interpolate import interp2d
from matplotlib.widgets import Cursor

import tkinter # Tkinter

from  matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


##############################################################################################################
# CONSTANTS
##############################################################################################################

working_path = 'D:\\Research_analysis\\Projects\\2024_JET\\Lab_comparison_test\\Photo_to_spectrum\\Photos_spectra_comparison\\'
input_path = working_path + 'Input\\'
averaged_path = working_path + 'Averaged\\'
output_path = working_path + 'Output\\'
spectrum_path = working_path + 'Spectra\\'

##############################################################################################################
# DEFINITIONS
# Definitions are mostly structured in the order in which they are called at first.
##############################################################################################################

def main_program():
    # create folders if missing
    create_folder(output_path)
    create_folder(averaged_path)
    
    # average input photos and return the array
    average_array, identificator = prepare_photos()
    
    # assumes square image
    if (identificator == use_sample):
        process_photo(average_array, identificator)
    
    return

##############################################################################################################
# Preparation phase
##############################################################################################################

def prepare_photos():
    # Get files
    input_files = get_folder_files(input_path)
    averaged_files = get_folder_files(averaged_path)
    
    # Get output file identificator
    name = input_files[0] # identificator_0025.tif
    pattern = r'^(.+?)_\d+?\.tif$' # (any characters), _, digits, ., tif
    identificator = re.search(pattern, name)
    if identificator: # match found, extract value
        identificator = identificator.group(1)
    else: # no match
        identificator = ''
    
    exif_data = None
    
    averaged_name = 'average_' + identificator + '.tif'
    if averaged_name in averaged_files: # averaging done previously
        average_array, exif_data = load_photo(averaged_path + averaged_name)
    
    # Average all photos in input folder and save averaged photo into averaged folder
    else:
        average_array, exif_data = average_photos(input_files)
        output_averaged_photo(average_array, output_path, identificator, exif_data)
    
    # Convert spectra to photos for GIMP or something else
    spectra_to_photos()
    
    return average_array, identificator


def average_photos(input_files):
    
    # Initialize averaging variables
    count = len(input_files)
    sum = None
    
    # iterate over files
    for filename in input_files:
        imArray, exif_data = load_photo(input_path + filename)
        
        # sum arrays
        if sum is None: # initialize
            sum = imArray
        else:  # add
            sum += imArray
    
    average = sum / count
    return average, exif_data

def load_photo(filepath):
    image = PILImage.open(filepath)
    #im.show()
    #exif_data = image.info['exif'] # save exif data, so image can be opened with Sophi nXt
    exif_data = image.getexif() # save exif data, so image can be opened with Sophi nXt
    # TODO: get exif data (especially description (or comment) part) and save with new image
    
    imArray = np.array(image)
    return imArray, exif_data


def output_averaged_photo(average_array, output_path, identificator, exif_data = {}):
    #for k, v in exif_data.items():
    #    print("Tag", k, "Value", v)  # Tag 274 Value 2
    
    image = PILImage.fromarray(average_array)
    image.save(averaged_path + 'average_' + identificator + '.tif')#, exif = exif_data) 
    
def spectra_to_photos():
    files = get_folder_files(spectrum_path)
    
    for file in files:
        if file.endswith(".dat") or file.endswith(".csv"):
            #spectrum = np.fromfile(spectrum_path + file, dtype=float, sep=' ') # Todo: more separators
            spectrum = np.loadtxt(spectrum_path + file, delimiter=' ', usecols = (1))
            #load_spectrum(spectrum_path + file)
            
            # Save image
            image = PILImage.fromarray(spectrum)
            #image.show()
            image.save(spectrum_path + file + '.tif') # TODO: strip file extension first
            

def load_spectrum(filepath):
    array = np.fromfile(filepath, dtype=float, sep=' ', like = np.empty((39303, 2)))
    return



##############################################################################################################
# Processing phase
##############################################################################################################

def process_photo(photo_array, identificator):
    
    # negative values to 0
    photo_array[photo_array < 0] = 0
    
    
    # Load calibration data if it exists
    calibr_data = load_calibration_data()
    
    #calibration_window(photo_array)
    
    tkinter_master = tkinter.Tk()
    try:
        window = calibration_window(tkinter_master, photo_array, calibration_dict = calibr_data)
        tkinter_master.mainloop() # run tkinter (only once)
    finally:
        # When code crashes or debugging is stopped then the tkinter object isn't destroyed in Spyder
        # This isn't probably an issue in the final program version.
        try: # another try in case user closed the window manually and tkinter_master doesn't exist
            tkinter_master.destroy()
        except:
            pass
    
    np.savetxt( output_path + 'image_array.csv', photo_array, delimiter = ',')
    #draw_pyplot(photo_array, identificator)
    
    return

def load_calibration_data():
    data = None
    
    output_files = get_folder_files(output_path)
    
    if 'calibration_data.json' in output_files: # calibration done previously
        with open(output_path + 'calibration_data.json', 'r') as file:
            try:
                data = json.load(file)
            except: # e.g. file is empty
                pass
    
    return data


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



# Window to calibrate the echellogram to spectrum curves
class calibration_window():
    def __init__(self, parent, photo_array, calibration_dict = None):
        self.root = parent
        self.photo_array = photo_array
        
        #self.root.winfo_width()
        #self.root.winfo_height()
        #self.root.winfo_screenwidth()
        #self.root.winfo_screenheight()
        
        op_sys_toolbar_height = 0.07 #75 px for 1080p
        self.root.maxsize(self.root.winfo_screenwidth(), self.root.winfo_screenheight() - round(self.root.winfo_screenheight() * op_sys_toolbar_height))
        
        self.frame_spectrum = tkinter.Frame(self.root)#, height = self.root.winfo_height() / 3)
        self.frame_buttons_image = tkinter.Frame(self.root)
        self.frame_buttons = tkinter.Frame(self.frame_buttons_image)
        self.frame_image = tkinter.Frame(self.frame_buttons_image)
        
        self.program_mode = None
        self.selected_order = None
        self.calib_data = {} # dictionary to store three calibration things (start and end vertical, nr of horizontal)
        
        self.load_data(calibration_dict)
        
        
        
        # Create buttons
        self.button_reset_btn = tkinter.Button(self.frame_buttons, text = "Reset mode", command = self.reset_mode)
        self.vertical_mode_btn = tkinter.Button(self.frame_buttons, text = "Vertical curve edit mode", command = self.bounds_mode)
        self.orders_mode_btn = tkinter.Button(self.frame_buttons, text = "Diffr order curve edit mode", command = self.orders_mode)
        self.button_save_btn = tkinter.Button(self.frame_buttons, text = "Save data", command = self.save_data)
        
        self.add_order_btn = tkinter.Button(self.frame_buttons, text = "Add diffr order", command = self.add_order)
        #self.add_point_left = tkinter.Button(self.frame_buttons, text = "Add point to left vertical", command = self.add_calibr_point_left)
        #self.add_point_right = tkinter.Button(self.frame_buttons, text = "Add point to right vertical", command = self.add_calibr_point_right)
        #self.add_point_order = tkinter.Button(self.frame_buttons, text = "Add point to order", command = self.add_calibr_point_order)
        self.spectrum_log_btn = tkinter.Button(self.frame_buttons, text = "Toggle spectrum log scale", command = self.spectrum_toggle_log)
        self.image_int_down_btn = tkinter.Button(self.frame_buttons, text = "Lower colorbar max", command = self.image_int_down)
        self.image_int_up_btn = tkinter.Button(self.frame_buttons, text = "Raise colorbar max", command = self.image_int_up)
        self.image_int_min_down_btn = tkinter.Button(self.frame_buttons, text = "Lower colorbar min", command = self.image_int_min_down)
        self.image_int_min_up_btn = tkinter.Button(self.frame_buttons, text = "Raise colorbar min", command = self.image_int_min_up)
        
        # Show buttons
        self.button_save_btn.pack()
        self.button_reset_btn.pack()
        self.vertical_mode_btn.pack()
        self.orders_mode_btn.pack()
        self.add_order_btn.pack()
        self.spectrum_log_btn.pack()
        self.image_int_down_btn.pack()
        self.image_int_up_btn.pack()
        self.image_int_min_down_btn.pack()
        self.image_int_min_up_btn.pack()
        
        
        # some tkinter classes like BaseWidget() assume tkinter is imported as "tk" 
        #self.tk = tkinter
        
        self.root.title("Echellogram calibration")
        #self.root.configure(background="yellow")
        self.root.minsize(600, 600)
        #self.root.maxsize(500, 500)
        #self.root.geometry("300x300+50+50")
        
        #self.root.update()
        
        #self.submit = tkinter.Button(self, text="Submit", command = self.create_label)
        
        #tkinter.Label(root, image=image).pack()
        self.draw_plot(identificator = '')
        
        self.draw_calibr_curves()
        
        self.draw_spectrum()
        
        # Draw things
        self.frame_buttons_image.pack(fill='both', expand=1, side = 'top')
        self.frame_spectrum.pack(fill='both', expand=1, side = 'top')
        self.frame_buttons.pack(side = 'left')
        self.frame_image.pack(side = 'left')
        
    
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
            self.calib_data['orders'].sort(key=lambda x: x.avg_y, reverse=True)
    
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
    
    
    
    def reset_mode(self):
        self.program_mode = None
        self.selected_order = None
    def bounds_mode(self):
        self.program_mode = 'bounds'
        self.selected_order = None
    def orders_mode(self):
        self.program_mode = 'orders'
        self.selected_order = None
    
    
    def add_order(self):
        diffr_order = order_points(self.photo_array.shape[0])
        self.calib_data['orders'] += [diffr_order]
        self.plot_order(diffr_order)
        
        # sort orders by average y values
        #self.calib_data['orders'].sort(key=lambda x: x.avg_y, reverse=True)
        
        self.selected_order = diffr_order
        self.best_order_idx = len(self.calib_data['orders']) - 1
        
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
        
        #self.update_order_curves()
        self.update_spectrum()
        
        xdata = self.spectrum_curve.get_xdata()
        self.spectrum_ax.set_xlim(min(xdata), max(xdata))
        
        # Draw plot again and wait for drawing to finish
        self.canvas_spectrum.draw()
        self.canvas_spectrum.flush_events() 
    
    def spectrum_toggle_log(self):
        current_scale = self.spectrum_ax.get_yscale()
        if current_scale == 'symlog':
            self.spectrum_ax.set_yscale('linear')
        else:
            self.spectrum_ax.set_yscale('symlog') # symlog for 0-values
        
        # Draw plot again and wait for drawing to finish
        self.canvas_spectrum.draw()
        self.canvas_spectrum.flush_events() 
        
    def image_int_down(self):
        old_min = self.cbar.vmin
        old_max = self.cbar.vmax
        self.image.set_clim(old_min, old_max / 5)
        
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
    
    def image_int_up(self):
        old_min = self.cbar.vmin
        old_max = self.cbar.vmax
        self.image.set_clim(old_min, old_max * 5)
        
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
    
    def image_int_min_down(self):
        old_min = self.cbar.vmin
        old_max = self.cbar.vmax
        self.image.set_clim(old_min / 5, old_max)
        
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
    
    def image_int_min_up(self):
        old_min = self.cbar.vmin
        old_max = self.cbar.vmax
        self.image.set_clim(old_min * 5, old_max)
        
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
    
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
            
            # sort orders by average y values
            #self.calib_data['orders'].sort(key=lambda x: x.avg_y, reverse=True)
            
            self.update_order_curves()
            
        
        # Redraw plot
        if not (self.program_mode is None):
            self.update_spectrum()
    
    
    def draw_plot(self, identificator = ''):
        self.photo_fig = plt.figure()
        
        # Attach figure to the Tkinter frame
        self.canvas = FigureCanvasTkAgg(self.photo_fig, self.frame_image)
        toolbar = NavigationToolbar2Tk(self.canvas, self.frame_image)
        toolbar.update()
        self.canvas._tkcanvas.pack(fill='both', expand=1, side = 'left')
        
        self.canvas.mpl_connect('button_press_event', self.plot_click_callback)
        #matplotlib.backend_bases.MouseEvent
        
        #draw_pyplot(photo_array, root_fig = self.photo_fig, identificator = '')
        self.format_plot()
        
    
    def format_plot(self):
        
        # x-axis to the top of the image
        self.plot_ax = self.photo_fig.gca()
        self.plot_ax.xaxis.tick_top()
        plt.tight_layout()
        
        # Show image and colorbar
        self.image = plt.imshow(self.photo_array, norm = 'log')
        self.cbar = plt.colorbar()
    
    def draw_calibr_curves(self):
        # Get line data
        y_values = np.arange(self.photo_array.shape[0])
        left_curve_array, self.left_poly_coefs = get_polynomial_points(self.calib_data['start'], self.photo_array.shape[0], flip = True)
        right_curve_array, self.right_poly_coefs = get_polynomial_points(self.calib_data['end'], self.photo_array.shape[0], flip = True)
        
        # Plot curves
        self.start_curve, = self.plot_ax.plot(left_curve_array, y_values, 'tab:orange')
        self.end_curve, = self.plot_ax.plot(right_curve_array, y_values, 'tab:orange')
        
        # Plot calibration points
        self.start_curve_points, = self.plot_ax.plot(self.calib_data['start'].xlist, self.calib_data['start'].ylist, color = 'k', marker = 'o', linestyle = '')
        self.end_curve_points, = self.plot_ax.plot(self.calib_data['end'].xlist, self.calib_data['end'].ylist, color = 'k', marker = 'o', linestyle = '')
        
        self.plot_orders()
    
        
    def plot_orders(self):
        self.order_plot_curves = []
        self.order_plot_points = []
        self.order_poly_coefs = []
        
        # Iterate over diffraction orders
        x_values = np.arange(self.photo_array.shape[0])
        for order in self.calib_data['orders']:
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
        curve_points, = self.plot_ax.plot(order.xlist, order.ylist, color = 'r', marker = 'o', linestyle = '')
        self.order_plot_points.append(curve_points)
        
    
    
    # Redraw vertical calibration curves
    def update_vertical_curves(self):
        
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
        
        # Draw plot again and wait for drawing to finish
        self.canvas.draw()
        self.canvas.flush_events() 
    
    
    # Redraw horizontal calibration curves
    def update_order_curves(self):
        # Iterate over diffraction orders
        for idx in range(len(self.calib_data['orders'])):
            order = self.calib_data['orders'][idx]
            
            # points
            self.order_plot_points[idx].set_xdata(order.xlist)
            self.order_plot_points[idx].set_ydata(order.ylist)
            
            # curve
            curve_array, poly_coefs = get_polynomial_points(order, self.photo_array.shape[0])
            self.order_poly_coefs[idx] = poly_coefs
            self.order_plot_curves[idx].set_ydata(curve_array)
            
            # Draw plot again and wait for drawing to finish
            self.canvas.draw()
            self.canvas.flush_events() 
    
    
    def draw_spectrum(self):
        self.spectrum_fig = plt.figure()
        
        # Attach figure to the Tkinter frame
        self.canvas_spectrum = FigureCanvasTkAgg(self.spectrum_fig, self.frame_spectrum)
        toolbar = NavigationToolbar2Tk(self.canvas_spectrum, self.frame_spectrum)
        toolbar.update()
        self.canvas_spectrum._tkcanvas.pack(fill='both', expand=1, side = 'top')
        
        self.spectrum_ax = self.spectrum_fig.gca()
        self.format_spectrum()
        
        # Plot curve
        x_values, z_values = self.get_order_spectrum()
        self.spectrum_curve, = self.spectrum_ax.plot(x_values, z_values, 'tab:pink')
        
    def format_spectrum(self):
        #self.spectrum_ax.autoscale()
        pass
    
    
    def update_spectrum(self):
        
        x_values, z_values = self.get_order_spectrum()
        
        # Plot curve
        self.spectrum_curve.set_xdata(x_values)
        self.spectrum_curve.set_ydata(z_values)
        
        # rescale to fit
        #self.spectrum_ax.set_xlim(min(x_values), max(x_values))
        self.spectrum_ax.set_ylim(min(z_values), max(z_values))
        
        # Draw plot again and wait for drawing to finish
        self.canvas_spectrum.draw()
        self.canvas_spectrum.flush_events() 
    
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
        x2, y2 = sort_connected_arrays(x2, y2)
        x3, y3 = sort_connected_arrays(x3, y3)
        '''
        nr_pixels = self.photo_array.shape[0]
        
        # Compile z-data from the photo with all diffraction orders
        photo_pixels = np.arange(nr_pixels)
        for idx in range(len(self.calib_data['orders'])):
            #order = self.calib_data['orders'][idx]
            
            #curve_x = photo_pixels
            curve_x = self.order_plot_curves[idx].get_xdata()
            curve_y = self.order_plot_curves[idx].get_ydata()
            
            # sort the arrays in increasing x value
            #curve_x, curve_y = sort_connected_arrays(curve_x, curve_y)
            
            
            # Interpolate and get y-values of the calibration curve at every pixel
            #y_pixels = np.interp(photo_pixels, curve_x, curve_y) # pixel values are already calculated by the polynomial
            #y_pixels = y_pixels.astype(np.int32) # convert to same format as x-values
            y_pixels = curve_y.astype(np.int32) # convert to same format as x-values
            
            left_curve = self.start_curve.get_xdata()
            right_curve = self.end_curve.get_xdata()
            
            # Get bounds for this diffraction order
            x_left, x_right = get_order_bounds(nr_pixels, curve_x, curve_y, left_curve, right_curve)
            #x_left = get_polynomial_intersection(self.order_poly_coefs[idx], self.left_poly_coefs, nr_pixels, is_left = True)
            #x_right = get_polynomial_intersection(self.order_poly_coefs[idx], self.right_poly_coefs, nr_pixels, is_left = False)
            
            #print(x_left, x_right)
            
            # get z-values corresponding to the interpolated pixels on the order curve
            for idx in range(len(photo_pixels)): # TODO: add width integration
                
                # Save px only if it's between left and right calibration curves
                x_value = curve_x[idx]
                y_value = y_pixels[idx]
                
                if x_right is None:
                    pass
                
                #if self.point_in_range(x_value, curve_x, curve_y, x2, y2, x3, y3): # intersection function is way too slow
                if x_left <= x_value <= x_right:
                    if x_values is None:
                        x_values = [0]
                    else:
                        x_values.append(x_values[-1] + 1) # increment by 1
                    
                    integral = integrate_order_width(self.photo_array, x_value, y_value, width = integral_width)
                    z_values.append(integral) # x and y have to be switched (somewhy)
                
        return x_values, z_values
    
    
    
    
    def point_in_range(self, x_point, x1, y1, x2, y2, x3, y3):
        
        x_start,_ = intersection(x1,y1,x2,y2) # intersection function is way too slow
        x_start = x_start[0]
        
        x_end,_ = intersection(x1,y1,x3,y3)
        x_end = x_end[0]
        
        if x_start <= x_point <= x_end:
            return True
        return False
    
    def create_label(self):
        # Create two labels
        tkinter.Label(self.root, text="Nothing will work unless you do.").pack()
        tkinter.Label(self.root, text="- Maya Angelou").pack()
    
    def increase(self):
        pass



# Sum pixels around the order
def integrate_order_width(photo_array, x_pixel, y_pixel, width = 1):
    integral = 0
    for y_idx in range(y_pixel - round(width / 2), y_pixel + round(width / 2)):
        integral += photo_array[y_idx, x_pixel] # x and y have to be switched (somewhy)
    
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
def sort_connected_arrays(arr1, arr2):
    sort_idx = np.argsort(arr1)
    arr1 = arr1[sort_idx]
    arr2 = arr2[sort_idx]
    return arr1, arr2

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
def get_folder_files(path):
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return onlyfiles

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