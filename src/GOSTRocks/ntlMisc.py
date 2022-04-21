import sys, os, inspect, logging, json
import rasterio, boto3

import pandas as pd
import geopandas as gpd
import numpy as np

def map_viirs(cur_file, out_file='', class_bins = [-10,0.5,1,2,3,5,10,15,20,30,40,50], text_x=0, text_y=5, dpi=100):
    ''' create map of viirs data
    
    INPUT
        cur_file [string] - path to input geotif
        [optional] out_file [string] - path to create output image
        [optional] class_bins [list numbers] - breaks for applying colour ramp
        [optional] text_x [int] - position on map to position year text (left to right)
        [optional] text_y [int] - position on map to position year text (top to bottom)
    '''
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

def find_monthly_ntl(bucket='wbgdecinternal-ntl', prefix='NTL/VIIRS/Annual/VIIRS_ANNUAL_EOG'):
    ''' Get list of NTL files in AWS
    
    INPUT
    bucket [string, optional] - bucket to search for imagery
    prefix [string, optional] - prefix storing images. Default is annual composites from EOG, choose "NTL/VIIRS" for monthly
    
    RETURNS
    [pandas dataframe]
    '''

    s3client = boto3.client('s3')
    
    # Loop through the S3 bucket and get all the keys for files that are .tif 
    more_results = True
    try:
        del(token)
    except:
        pass
    loops = 0
    good_res = []
    while more_results:
        print(f"Completed loop: {loops}")
        if loops > 0:
            objects = s3client.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=token)
        else:
            objects = s3client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        more_results = objects['IsTruncated']
        if more_results:
            token = objects['NextContinuationToken']
        loops += 1
        for res in objects['Contents']:
            if res['Key'].endswith('.tif'):
                good_res.append(f"s3://{bucket}/{res['Key']}")
    
    return(good_res)

