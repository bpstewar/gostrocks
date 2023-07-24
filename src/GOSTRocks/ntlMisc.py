import sys, os, inspect
import rasterio

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

curPath = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if not curPath in sys.path:
    sys.path.append(curPath)

from dataMisc import aws_search_ntl
from misc import tPrint
import rasterMisc as rMisc

def map_viirs(cur_file, out_file='', class_bins = [-10,0.5,1,2,3,5,10,15,20,30,40,50], text_x=0, text_y=5, dpi=100):
    """Map VIIRS nighttime lights imagery, optionally create output image

    :param cur_file: path to input geotiff
    :type cur_file: string
    :param out_file: path to create output image, defaults to '' which does not create a file
    :type out_file: str, optional
    :param class_bins: breaks for applying colour ramp, defaults to [-10,0.5,1,2,3,5,10,15,20,30,40,50]
    :type class_bins: list, optional
    :param text_x: position on map to position year text (left to right), defaults to 0
    :type text_x: int, optional
    :param text_y: position on map to position year text (top to bottom), defaults to 5
    :type text_y: int, optional
    :param dpi: dotes per inch for output image, defaults to 100
    :type dpi: int, optional
    """
    # extract the year from the file name
    year = cur_file.split("_")[-1][:4]
    
    # Open the VIIRS data and reclassify 
    inR = rasterio.open(cur_file)
    inD = inR.read() 
    inC = xr.apply_ufunc(np.digitize,inD,class_bins)

    # Plot the figure, remove grid and ticks
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    ### TODO: add the year to the map, may need to experiment with the location depend on geography
    ax.text(text_x, text_y, year, fontsize=40, color='white')

    #plt.margins(0,0)
    if out_file != '':
        #plt.imsave(out_file, inC[0,:,:], cmap=plt.get_cmap('magma'))
        plt.imshow(inC[0,:,:], cmap=plt.get_cmap('magma'))
        fig.savefig(out_file, dpi=dpi, bbox_inches='tight', pad_inches=0)
    else:
        # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        plt.imshow(inC[0,:,:], cmap=plt.get_cmap('magma'))

    
def run_zonal(inD, ntl_files=[], minval=0.1, verbose=False, calc_sd=True):
    """ Run zonal statistics against a series of nighttime lights files

    :param inD: input geopandas dataframe in which to summarize results
    :type inD: gpd.GeoDataFrames
    :param ntl_files: list of ntl files to summarize, defaults to [] which will search for all files in the s3 bucket using datMisc.aws_search_ntl()
    :type ntl_files: list, optional
    :param minval: Minimum value to summarize in nighttime lights, defaults to 0.1 which means all values below this become 0
    :type minval: float, optional
    :param verbose: print additional information, defaults to False
    :type verbose: bool, optional
    :param calc_sd: _description_, defaults to True
    :type calc_sd: bool, optional
    """
    
    ''' run zonal stats on all ntl files
    INPUT 
        inD [geopandas dataframe]
        
    RETURNS
        pandas dataframe
    '''
    if len(ntl_files) == 0:
        ntl_files = aws_search_ntl()
        
    for ntl_file in ntl_files:
        name = ntl_file.split("/")[-1].split("_")[2][:8]
        if verbose:
            tPrint(name)
        inR = rasterio.open(ntl_file)
        ntl_res = rMisc.zonalStats(inD, inR, minVal=minval, calc_sd=calc_sd) 
        out_cols = ['SUM','MIN','MAX','MEAN']
        if calc_sd:
            out_cols.append("SD")
        ntl_df = pd.DataFrame(ntl_res, columns=out_cols)
        inD[f'ntl_{name}_SUM'] = ntl_df['SUM']
    return(inD)
    

