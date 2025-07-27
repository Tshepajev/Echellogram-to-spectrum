# Author: Jasper Ristkok

# The script reads Sophy nXt .tif files and exports the data to .csv files.
# The data contains Echellogram (2D intensity matrix) and metadata.

imgpath = r'E:\Research_analysis\2024.09 JET\Lab_comparison_test\Photo_to_spectrum\Test\492_2X8_R7C3C7_0001 2X08 - R7 C3-C7.tif'
outputpath = r'E:\Nextcloud sync\Data_processing\Projects\2024_09_JET\Lab_comparison_test\Photo_to_spectrum\Analysis\aryx export\\'


# Must be installed if IDE doesn't have these installed already
import ltbfiles # (https://pypi.org/project/ltbfiles/) - https://gitlab.com/ltb_berlin/ltb_files
import numpy # (https://pypi.org/project/numpy/)


def main():
    extract_img_data(imgpath)
    print('Program finished. Outputs written to\n' + outputpath)

# Read an image (.tif) and print the metadata
def extract_img_data(filepath):
    # [2D int, metadata]
    imgdata = ltbfiles.read_ltb_tiff(filepath) 
    intensity_array = imgdata[0]
    metadata = imgdata[1]
    
    print('Image metadata: ' + str(metadata))
    
    with open(outputpath + 'Extracted_tif_metadata.txt', 'w') as file:
        file.write(str(metadata))
    
    #numpy.savetxt(outputpath + 'Extracted_tif_metadata.csv', str(metadata), fmt = '%.5e', delimiter = ',', comments = '')
    numpy.savetxt(outputpath + 'Extracted_tif_matrix.csv', intensity_array, fmt = '%.10e', delimiter = ',', comments = '')
    

# Run the code
main()