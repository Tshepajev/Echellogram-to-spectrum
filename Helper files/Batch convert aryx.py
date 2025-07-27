# Author: Jasper Ristkok

# The script reads Sophy nXt .aryx files and exports all the spectra in the folder to .txt files.
# The output file is the same as if you exported the .aryx file in Sophi nXt.
# The data contains wavelengths and intensities separated with a semicolon ";".

# TODO: format output string for less file size. Wavelengths with 9 significant digits (general number)
# and intensity as general number (seems to be integer)

# https://gitlab.com/ltb_berlin/ltb_files

# If path is None then path is where you executed the script (script location), otherwise the string you wrote
path = None
#path = r'E:\Research_analysis\2024.09 JET\Lab_comparison_test\Raw data\20_14WLH_AR3C1\ARYELLE SPECTROMETER\\'


# These are included in Python Standard Library (comes with Python)
import os
import platform

# Must be installed if IDE doesn't have these installed already
import ltbfiles # (https://pypi.org/project/ltbfiles/)
import numpy # (https://pypi.org/project/numpy/)


def main():
    
    # Code is executed from folder with spectra
    if path is None:
        get_path()
    
    batch_convert_aryx(path)
    print('Exporting finished.')


# If you didn't provide a path in the beginning then use path
# as the folder where the script (.py file) is executed from
def get_path():
    global path
    
    # Get OS name for determining path symbol
    if platform.system() == 'Windows':
        system_path_symbol = '\\'
    else:
        system_path_symbol = '/' # Mac, Linux
    
    # Get folder where .py file was executed from
    path = os.getcwd() # 'd:\\github\\echellogram-to-spectrum\\Helper files'
    path += system_path_symbol
    

# Read a spectrum (.aryx) and save the data into a separate .txt file
def batch_convert_aryx(path):
    print('Starting conversion.')
    
    # Can't use ltbfiles.load_folder() because it doesn't give filenames
    #spectra_list = ltbfiles.load_folder(path, interpolate = False, extensions = ['.ary', '.aryx'])
    
    # Get all spectra in the folder and iterate over the spectra
    iteration = 0
    filenames = get_files_in_folder(path, ['.ary', '.aryx'])
    for filestring in filenames:
        filestring = str(filestring)
        
        # Write progress
        if iteration % 20 == 0:
            print('Processing ' + filestring)
        
        # Catch potential errors
        try:
            process_spectrum(filestring)
        except Exception as e:
            print('Error with converting ' + filestring)
            print(e)
        finally:
            iteration += 1
        

# Get all filenames in folder with the corresponding extensions
def get_files_in_folder(path, extensions):
    correct_files = []
    
    # Get files, not directories
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
    # Get only the files with the correct extensions
    for filename in onlyfiles:
        _, ext = os.path.splitext(filename)
        if ext in extensions:
            correct_files.append(filename)
    
    return correct_files

# Extract data from .aryx file and export it to .txt.
def process_spectrum(filestring):
    filename, ext = os.path.splitext(filestring)
        
    # [int, wavelength, order_nr, metadata] if using as a list
    if ext == '.ary':
        spectrum = ltbfiles.read_ltb_ary(path + filestring, sort_wl = True) # sort or not
    elif ext == '.aryx':
        spectrum = ltbfiles.read_ltb_aryx(path + filestring, sort_wl = True) # sort or not
    
    wavelength = spectrum.x
    intensity = spectrum.Y
    
    wavelength, intensity = remove_overlapping_pixels(wavelength, intensity)
    
    array = numpy.empty([len(intensity), 2])
    array[:,0] = wavelength
    array[:,1] = intensity
    
    numpy.savetxt(path + filename + '.txt', array, fmt = '%.8e', delimiter = ';', comments = '')
    

# If two pixels have the same wavelength then one is deleted
def remove_overlapping_pixels(wavelength, intensity):
    wavelength, ind = numpy.unique(wavelength, True)
    intensity = intensity[ind]
    return wavelength, intensity

# Run the code
main()