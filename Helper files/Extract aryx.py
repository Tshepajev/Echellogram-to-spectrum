# Author: Jasper Ristkok

# The script reads Sophy nXt .aryx files and exports the data to .csv files.
# The data contains wavelengths, intensities, order nrs and metadata (at the end of the file).
# The script also extracts aif data (orders info) from aryx that can be used to create Vertical_points.csv

# https://gitlab.com/ltb_berlin/ltb_files
import ltbfiles
import numpy
import zipfile

#aryxpath = r'E:\Research_analysis\2024.09 JET\Lab_comparison_test\Raw data\test_12_08_2024\DHLamp__0001.aryx'
#aryxpath = r'E:\Research_analysis\2024.09 JET\Lab_comparison_test\Raw data\492 - 2X08 - R7 C3-C7 Be ref\492_2X8_R7C3C7_0001 2X08 - R7 C3-C7.aryx'
aryxpath = r'E:\Research_analysis\2024.09 JET\Lab_comparison_test\Raw data\20_14WLH_AR3C1\ARYELLE SPECTROMETER\Aryelle_0001.aryx'

outputpath = r'E:\Nextcloud sync\Data_processing\Projects\2024_09_JET\Lab_comparison_test\Photo_to_spectrum\Analysis\aryx export\\'

def main():
    extract_aryx_spectrum(aryxpath)
    extract_aif_data(aryxpath)
    
    print('Program finished. Outputs written to\n' + outputpath)

# Read a spectrum (.aryx) and save the data into a separate .csv file
def extract_aryx_spectrum(filepath):
    # [int, wavelength, order_nr, metadata]
    spectrum = ltbfiles.read_ltb_aryx(filepath, sort_wl = False) # sort or not
    
    spectrum = remove_overlapping_pixels(spectrum)
    
    intensity = spectrum.Y
    wavelength = spectrum.x
    order_nr = spectrum.o
    metadata = spectrum.head
    
    #print('Spectrum metadata: ' + str(wavelength))
    
    array = numpy.empty([len(intensity), 3])
    array[:,0] = wavelength
    array[:,1] = intensity
    array[:,2] = order_nr
    
    headers = 'wavelength(nm),intensity,order_nr'
    numpy.savetxt(outputpath + 'Extracted_spectrum.csv', array, fmt = '%.10e', delimiter = ',', header = headers, footer = str(metadata), comments = '')

# If two pixels have the same wavelength then one is deleted
def remove_overlapping_pixels(spectrum):
    spectrum.x, ind = numpy.unique(spectrum.x, True)
    spectrum.o = spectrum.o[ind]
    spectrum.Y = spectrum.Y[ind]
    return spectrum

# https://gitlab.com/ltb_berlin/ltb_files
# https://ltb_berlin.gitlab.io/ltb_files/ltbfiles.html
# Separate function wasn't included to extract aif data, so this is almost copy-paste    
def extract_aif_data(filepath):
    
    # How aif file in aryx compressed folder is encoded
    AIF_DTYPE_ARYX = numpy.dtype([('indLow', numpy.int32),
                      ('indHigh', numpy.int32),
                      ('order', numpy.int16),
                      ('lowPix', numpy.int16),
                      ('highPix', numpy.int16),
                      ('foo', numpy.int16),
                      ('lowWave', numpy.float64),
                      ('highWave', numpy.float64)])
    
    
    # Read aryx compressed contents
    with zipfile.ZipFile(filepath) as f_zip:
        file_list = f_zip.namelist()
        
        # Iterate over files in the compressed folder
        for i_file in file_list:
            if i_file.endswith('~aif'):
                aif = f_zip.read(i_file)
                orders_info = numpy.frombuffer(aif, AIF_DTYPE_ARYX)
    
    #print(orders_info)
                
    headers = 'spectrum start px (unsorted),spectrum end px (unsorted),order nr,image start px,image end px,nothing,start wavelength (nm),end wavelength (nm)'
    numpy.savetxt(outputpath + 'Extracted_orders.csv', orders_info, fmt = '%.5e', delimiter = ',', header = headers, comments = '')
    

# Run the code
main()