################################################################################
# Miscellaneous Raster functions
# Benjamin Stewart, September 2018
# Purpose: Collect a number of useful raster functions in one place
################################################################################

import sys, os, inspect, logging, json
import rasterio, affine, pyproj

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rxr

from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import Counter
from shapely.geometry import box
from shapely import wkt
from affine import Affine
from rasterio import features
from rasterio.mask import mask
from rasterio.features import rasterize, MergeAlg
from rasterio.warp import reproject, Resampling
from rasterio import MemoryFile
from contextlib import contextmanager

import seaborn as sns
sns.set(font_scale=1.5, style="whitegrid")

curPath = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if not curPath in sys.path:
    sys.path.append(curPath)

from misc import tPrint

@contextmanager
def create_rasterio_inmemory(src, curData):
    '''Create a rasterio object in memory from a numpy array 
    
    :param dictionary src: - data dictionary describing the rasterio template i.e. - rasterio.open().profile
    :param numpy array curData: - numpy array from which to create rasterio object
    '''
    with MemoryFile() as memFile:
        with memFile.open(**src) as dataset:
            dataset.write(curData)
            del curData
        
        with memFile.open() as dataset:
            yield(dataset)
            
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

def clipRaster(inR, inD, outFile):
    ''' Clip input raster
    INPUT
    [rasterio object] inR = rasterio.open(r"Q:/GLOBAL/POP&DEMO/GHS/BETA/FULL/MT/MT.vrt")
    [geopandas object]inD = gpd.read_file(r"Q:\WORKINGPROJECTS\CityScan\Data\CityExtents\Conotou_AOI.shp")
    [string]          outFile = r"Q:\WORKINGPROJECTS\CityScan\Data\CityExtents\Conotou_AOI\MappingData\GHSL.tif"

    '''
    if inD.crs != inR.crs:
        inD = inD.to_crs(inR.crs)
        inD = inD.buffer(0)
    out_meta = inR.meta.copy()
    def getFeatures(gdf):
        #Function to parse features from GeoDataFrame in such a manner that rasterio wants them
        return [json.loads(gdf.to_json())['features'][0]['geometry']]
    tD = gpd.GeoDataFrame([[1]], geometry=[inD.unary_union])
    coords = getFeatures(tD)
    out_img, out_transform = mask(inR, shapes=coords, crop=True)
    out_meta.update({"driver": "GTiff",
                     "height": out_img.shape[1],
                     "width": out_img.shape[2],
                     "transform": out_transform})
    with rasterio.open(outFile, "w", **out_meta) as dest:
        dest.write(out_img)

def rasterizeDataFrame(inD, outFile, idField='', templateRaster='', templateMeta = '', nCells=0, res=0, mergeAlg="REPLACE", re_proj=False):
    ''' Convert input geopandas dataframe into a raster file
        inD = gpd.read_file(r"C:\Temp\TFRecord\Data\Training Data\test3_training.shp")
        templateRaster=r"C:\Temp\TFRecord\Data\Training Data\test3.tif"
        idField = 'ID2'
        outFile = templateRaster.replace(".tif", "_labels.tif")

    INPUT VARIABLES
        inD [geopandas DataFrame]
        outFile [string] - path for creating output file
    OPTIONAL
        idField [string] - field to rasterize, sets everything to 1 otherwise
        templateRaster [string] - raster upon which to base raster creation
        templateMeta [dictionary] - raster metadata used to create output raster
        nCells [number] - number of cells in width and height
        res [number] - resolution of output raster in units of the crs
        re_proj [Boolean] - option to reproject inD to templateRaster if CRS do not match
    RETURNS
        [dict of [dict] and [numpy array]] - metadata used to create output raster and burned raster values
    '''
    ###Parameter checking
    if nCells <=0 and res <=0 and templateRaster == '' and templateMeta =='':
        raise(ValueError("Must define one of nCells or res"))
    if nCells > 0 and res > 0 and templateRaster == ''  and templateMeta =='':
        raise(ValueError("Cannot define both nCells and res"))

    #Set VALUE field equal to idField
    inD['VALUE'] = 1
    inD['VALUE'] = inD['VALUE'].astype('int16')
    if idField != '':
        inD['VALUE'] = inD[idField]

    # set merge algorithm for overlapping features
    if mergeAlg == "REPLACE":
        mAlg = MergeAlg.replace
    elif mergeAlg == "ADD":
        mAlg = MergeAlg.add
    else:
        raise(ValueError("MergeAlg must be one of REPLACE or ADD"))
        
    if templateRaster != '':
        inR = rasterio.open(templateRaster)
        cMeta = inR.meta.copy()
        cMeta.update(count=1)
        nTransform = cMeta['transform']
        if inD.crs != inR.crs:
            if not re_proj:
                raise(ValueError("input CRS do not match: inD - %s, templateRaster - %s" % (inD.crs, inR.crs)))
            inD = inD.to_crs(inR.crs)
    elif templateMeta != '':
        cMeta = templateMeta
        nTransform = cMeta['transform']
        if inD.crs != cMeta['crs']:
            if not re_proj:
                raise(ValueError("input CRS do not match: inD - %s, templateRaster - %s" % (inD.crs, inR.crs)))
            inD = inD.to_crs(cMeta['crs'])
    else:
        bounds = inD.total_bounds
        if nCells > 0:
            cellWidth  = (bounds[2] - bounds[0]) / nCells
            cellHeight = ((bounds[3] - bounds[1]) / nCells) * -1
            height = nCells
            width = nCells
        if res > 0:
            cellWidth = res
            cellHeight = res
            height = int(round((bounds[3] - bounds[1]) / res))
            width =  int(round((bounds[2] - bounds[0]) / res))

        b = inD.total_bounds
        nTransform = rasterio.transform.from_bounds(b[0], b[1], b[2], b[3], width, height)
        if inD.crs.__class__ == pyproj.crs.crs.CRS:
            crs = {'init':'epsg:%s' % inD.crs.to_epsg()}
        else:
            crs = inD.crs
        print(crs)
        cMeta = {'count':1, 'crs': crs, 'dtype':inD['VALUE'].dtype, 'driver':'GTiff',
                 'transform':nTransform, 'height':height, 'width':width}
    shapes = ((row.geometry,row.VALUE) for idx, row in inD.iterrows())
    burned = features.rasterize(shapes=shapes, out_shape=(cMeta['height'], cMeta['width']), transform=nTransform, dtype=cMeta['dtype'], merge_alg=mAlg)
    try:
        with rasterio.open(outFile, 'w', **cMeta) as out:
            out.write_band(1, burned)
        return({'meta':cMeta, 'vals': burned})
    except:
        print("Error writing raster")
        return({'meta':cMeta, 'vals': burned})

def polygonizeArray(data, b, curRaster):
    '''
    Convert input array to a geodataframe defined by the boundary as the b object
    INPUT
    array [numpy array]
    curRaster [rasterio object] - template raster object
    RETURNS
    geopandas dataframe
    DEBUGGING
    window = ((float(lr[0]), float(ul[0])), (float(ul[1]), float(lr[1])+1))
    data = curRaster.read(bandNum, window=window, masked = True)
    outArray.to_csv("C:/temp/FUBAR.csv")
    '''
    #Calculate resolution of cells
    ll = curRaster.xy(*curRaster.index(*b[0:2]),"ll")
    xmin = ll[0]
    ymin = ll[1]
    xRes = curRaster.res[0]
    yRes = curRaster.res[1]
    crs = curRaster.crs
    #create a dataframe equal to the size of the array
    outArray = pd.DataFrame()
    outArray['id'] = list(range(0, (data.shape[0] * data.shape[1])))
    rowVals = []
    colVals = []
    actualvals = []
    for row in range(0,data.shape[0]):
        for col in range(0,data.shape[1]):
            rowVals.append(row)
            colVals.append(col)            
    outArray['row'] = rowVals
    outArray['col'] = colVals
    #Create a polygon covering each cell
    def getPolygon(x):
        llX = xmin + (xRes * x['col'])
        llY = ymin + (yRes * x['row'])
        A = "%s %s" % (llX, llY)
        B = "%s %s" % (llX, llY + yRes)
        C = "%s %s" % (llX + xRes, llY + yRes)
        D = "%s %s" % (llX + xRes, llY)
        return(wkt.loads("POLYGON((%s,%s,%s,%s,%s))" % (A,B,C,D,A)))
    outArray['geometry'] = outArray.apply(getPolygon, axis=1)
    outGeo = gpd.GeoDataFrame(outArray, geometry="geometry")
    outGeo.crs = crs
    return(outGeo)
    
def zonalStats(inShp, inRaster, bandNum=1, mask_A = None, reProj = False, minVal = '', maxVal = '',
                verbose=False , rastType='N', unqVals=[], weighted=False, allTouched=False):
    ''' Run zonal statistics against an input shapefile. Returns array of SUM, MIN, MAX, and MEAN

    INPUT VARIABLES
    inShp [string or geopandas object] - path to input shapefile
    inRaster [string or rasterio object] - path to input raster

    OPTIONAL
    bandNum [integer] - band in raster to analyze
    reProj [boolean] -  whether to reproject data to match, if not, raise an error
    minVal/maxVal [number] - if defined, will only calculate statistics on values above or below this number
    verbose [boolean] - whether to be loud with technical updates
    rastType [string N or C] - N is numeric and C is categorical. Categorical returns counts of numbers
    unqVals [array of numbers] - used in categorical zonal statistics, tabulates all these numbers, will report 0 counts
    mask_A [numpy boolean mask] - mask the desired band using an identical shape boolean mask. Useful for doing conditional zonal stats
    weighted [boolean] - apply weighted zonal calculations. This will determine the % overlap for each
        cell in the defined AOI. Will apply weights in calculations of numerical statistics
    
    RETURNS
    array of arrays, one for each feature in inShp
    '''
    if isinstance(inShp, str):
        inVector = gpd.read_file(inShp)
    else:
        inVector = inShp
    if isinstance(inRaster, str):
        curRaster = rasterio.open(inRaster, 'r')
    else:
        curRaster = inRaster

    # If mask is not none, apply mask
    if mask_A is not None:
        curRaster.write_mask(mask_A)

    outputData=[]
    if inVector.crs != curRaster.crs:
        if reProj:
            inVector = inVector.to_crs(curRaster.crs)
        else:
            raise ValueError("Input CRS do not match")
    fCount = 0
    tCount = len(inVector['geometry'])
    #generate bounding box geometry for raster bbox
    b = curRaster.bounds
    rBox = box(b[0], b[1], b[2], b[3])
    for idx, row in inVector.iterrows():
        geometry = row['geometry']
        fCount = fCount + 1
        try:
            #This test is used in case the geometry extends beyond the edge of the raster
            #   I think it is computationally heavy, but I don't know of an easier way to do it
            if not rBox.contains(geometry):
                geometry = geometry.intersection(rBox)
            try:
                if fCount % 1000 == 0 and verbose:
                    tPrint("Processing %s of %s" % (fCount, tCount) )
                # get pixel coordinates of the geometry's bounding box
                ul = curRaster.index(*geometry.bounds[0:2])
                lr = curRaster.index(*geometry.bounds[2:4])
                # read the subset of the data into a numpy array
                window = ((float(lr[0]), float(ul[0]+1)), (float(ul[1]), float(lr[1]+1)))

                if mask_A is not None:
                    data = curRaster.read(bandNum, window=window, masked = True)
                else:
                    data = curRaster.read(bandNum, window=window, masked = False)
                
                if weighted:
                    allTouched = True
                    #Create a grid of the input raster (data)
                    rGrid = polygonizeArray(data, geometry.bounds, curRaster)
                    #Clip the grid by the input geometry
                    rGrid['gArea'] = rGrid.area
                    rGrid['newArea'] = rGrid.intersection(geometry).area
                    #Store the percent overlap 
                    rGrid['w'] = rGrid['newArea']/rGrid['gArea']
                    newData = data
                    for idx, row in rGrid.iterrows():
                        newData[row['row'], row['col']] = data[row['row'], row['col']] * row['w']
                    data = newData
                    
                # create an affine transform for the subset data
                t = curRaster.transform
                shifted_affine = Affine(t.a, t.b, t.c+ul[1]*t.a, t.d, t.e, t.f+lr[0]*t.e)

                # rasterize the geometry
                mask = rasterize(
                    [(geometry, 0)],
                    out_shape=data.shape,
                    transform=shifted_affine,
                    fill=1,
                    all_touched=allTouched,
                    dtype=np.uint8)

                # create a masked numpy array
                masked_data = np.ma.array(data=data, mask=mask.astype(bool))
                if rastType == 'N':
                    if minVal != '' or maxVal != '':
                        if minVal != '':
                            masked_data = np.ma.masked_where(masked_data < minVal, masked_data)
                        if maxVal != '':
                            masked_data = np.ma.masked_where(masked_data > maxVal, masked_data)
                        if masked_data.count() > 0:
                            results = [np.nansum(masked_data), np.nanmin(masked_data), 
                                       np.nanmax(masked_data), np.nanmean(masked_data)]
                        else :
                            results = [-1, -1, -1, -1]
                    else:
                        results = [np.nansum(masked_data), np.nanmin(masked_data), 
                                   np.nanmax(masked_data), np.nanmean(masked_data)]
                if rastType == 'C':
                    if len(unqVals) > 0:
                        xx = dict(Counter(data.flatten()))
                        results = [xx.get(i, 0) for i in unqVals]
                    else:
                        results = np.unique(masked_data, return_counts=True)
                outputData.append(results)
            except Exception as e:
                if verbose:
                    print(e)
                if rastType == 'N':
                    outputData.append([-1, -1, -1, -1])
                else:
                    outputData.append([-1 for x in unqVals])
        except:
            print("Error processing %s" % fCount)
    return outputData

def standardizeInputRasters(inR1, inR2, inR1_outFile='', data_type="N"):
    ''' Standardize inR1 to inR2: changes crs, extent, and resolution.

    Inputs:
    inR1, inR2 [rasterio raster object]
    [optional] inR1_outFile [string] - output file for creating inR1 standardized to inR2
    [optional] data_type [string ['C','N']]
    
    Returns:
    [list] - [numpy array, raster metadata]
    '''
    if inR1.crs != inR2.crs:
        bounds = gpd.GeoDataFrame(pd.DataFrame([[1, box(*inR2.bounds)]], columns=["ID","geometry"]), geometry='geometry', crs=inR2.crs)
        bounds = bounds.to_crs(inR1.crs)
        b2 = bounds.total_bounds
        boxJSON = [{'type': 'Polygon', 'coordinates': [[[b2[0], b2[1]],[b2[0], b2[3]],[b2[2], b2[3]],[b2[2], b2[1]],[b2[0], b2[1]]]]}]
    else:
        b2 = inR2.bounds
        boxJSON = [{'type': 'Polygon', 'coordinates': [[[b2.left, b2.bottom],[b2.left, b2.top],[b2.right, b2.top],[b2.right, b2.bottom],[b2.left, b2.bottom]]]}]
    #Clip R1 to R2
    #Get JSON of bounding box
    out_img, out_transform = mask(inR1, boxJSON, crop=True)
    out_meta = inR1.meta.copy()
    #Re-scale resolution of R1 to R2
    newArr = np.empty(shape=(1, inR2.shape[0], inR2.shape[1]))
    
    if data_type == "N":
        resampling_type = Resampling.nearest
    elif data_type == "C":
        resampling_type = Resampling.cubic
    reproject(out_img, newArr, src_transform=out_transform, dst_transform=inR2.transform, src_crs=inR1.crs, dst_crs=inR2.crs, resampling=resampling_type)
    out_meta.update({"driver": "GTiff",
                     "height": newArr.shape[1],
                     "width": newArr.shape[2],
                     "transform": inR2.transform,
                     "crs": inR2.crs})
    if inR1_outFile != "":
        with rasterio.open(inR1_outFile, "w", **out_meta) as dest:
            dest.write(newArr.astype(out_meta['dtype']))
    return([newArr.astype(out_meta['dtype']), out_meta])

def jaccardIndex(inR1, inR2):
    '''Calculate the jaccard index on two binary input raster objects

    Reference: https://en.wikipedia.org/wiki/Jaccard_index

    Inputs:
    inR1/inR2[rasterio raster object] - these need to be the same size

    Returns:
    index [ float ]
    '''
    if inR1.shape != inR2.shape:
        print(inR1.shape)
        print(inR2.shape)
        raise ValueError("Shape of input rasters do not match")
    #Add the two rasters together and get the unique tabulation
    inC = inR1.read() + inR2.read()
    xx = np.unique(inC, return_counts=True)
    outDict = {}
    for itemIdx in range(0, len(xx[0])):
        outDict[xx[0][itemIdx]] = xx[1][itemIdx]

    #The resulting could have some weird numbers, but values 1 and 2 should be the focus.
    #   1 - Only one area defines it as urban
    #   2 - Both areas define cell as urban
    # Jaccard is ratio of 2 / 1+2
    try:
        jIdx = outDict[2] / float(outDict[2] + outDict[1])
        return jIdx
    except:
        return -1

def groupJaccard(oFile, sFiles):
    with open(oFile, 'w') as output:
        for bFile in sFiles:
            inR2 = rasterio.open(bFile)
            for cFile in sFiles:
                if bFile != cFile:
                    logging.info("Processing %s and %s" % (os.path.basename(bFile), os.path.basename(cFile)))
                    inR1 = rasterio.open(cFile)
                    curIndex = jaccardIndex(inR2, inR1)
                    output.write("%s,%s,%s\n" % (os.path.basename(bFile), os.path.basename(cFile), curIndex))

class zonalResult(object):
    def __init__(self, inputPath, fileType, fieldToCopy='ALL', fieldAction= 'REPLACE', fieldNames=''):
        '''
        INPUT
            inputPath [string] - path to the input zonal stats results
            fileTpe [string] - 'C' or 'N', for use in processing fields
            [optional] fieldToCopy [string] - field name type to extract from numerical fields, can be a regex expression
            [optional] fieldAction [string] - 'REPLACE' or 'JOIN'. REPLACE will replace output names with the variable
                fieldNames. JOIN will join fieldNames to each of the existing zonal results fields
            [optional] field_names [string] - either a list of field names to give to output, or prefix to join to each field               
        '''
        self.inputPath = inputPath
        self.inValues = pd.read_csv(inputPath)
        self.fileType = fileType
        #Extract specific fields
        self.inValues = self.inValues.drop(self.inValues.filter(regex="Unnamed").columns, axis=1)
        if fieldToCopy != 'ALL':
            #Search columns for defined field
            self.inValues = self.inValues.filter(regex="|".join(fieldToCopy))
        if fieldNames != '':
            if fieldAction == 'REPLACE':
                self.inValues.columns = fieldNames
            if fieldAction == 'JOIN':
                self.inValues.columns = ["%s_%s" % (fieldNames, c) for c in self.inValues.columns]
    def __str__(self):
        return("%s: %s" % (os.path.basename(self.inputPath), "|".join(self.inValues.columns)))

def runAllJaccard():
    inAI_file = r"Q:\WORKINGPROJECTS\Indonesia_Urbanization\UrbanComparison\ForBen\AI\d150t60pop50l.tif"
    inHD_file = r"Q:\WORKINGPROJECTS\Indonesia_Urbanization\UrbanComparison\ForBen\Clusters\hd_clusters.tif"
    inURB_file = r"Q:\WORKINGPROJECTS\Indonesia_Urbanization\UrbanComparison\ForBen\Clusters\urban_clusters.tif"
    inNTL1_file = r"Q:\WORKINGPROJECTS\Indonesia_Urbanization\UrbanComparison\ForBen\NTLs\NTL_Extents_80.tif"
    inNTL2_file = r"Q:\WORKINGPROJECTS\Indonesia_Urbanization\UrbanComparison\ForBen\NTLs\NTL_Extents_25.tif"
    inDuranton10 = r"Q:\WORKINGPROJECTS\Indonesia_Urbanization\UrbanComparison\ForBen\Duranton\Duranton_10_binary.tif"
    inDuranton7 = r"Q:\WORKINGPROJECTS\Indonesia_Urbanization\UrbanComparison\ForBen\Duranton\Duranton_7_binary.tif"
    allFiles = [inHD_file, inURB_file, inAI_file, inDuranton10, inDuranton7, inNTL2_file, inNTL1_file]

    outputFile = r"Q:\WORKINGPROJECTS\Indonesia_Urbanization\UrbanComparison\ForBen\Jaccard_Comparison.csv"
    baliShp = r"Q:\WORKINGPROJECTS\Indonesia_Urbanization\UrbanComparison\ForBen\Java_and_Bali.shp"
    baliGeom = gpd.read_file(baliShp)
    notBaliShp = r"Q:\WORKINGPROJECTS\Indonesia_Urbanization\UrbanComparison\ForBen\Not_Bali_norJAVA.shp"
    notBaliGeom = gpd.read_file(notBaliShp)

    #Standardize all input layers to the same baseline - inNTL1_file
    inR1 = rasterio.open(inNTL1_file)
    standardFiles = []

    logging.info("Standardizing input files")
    for inFile in allFiles:
        inR2 = rasterio.open(inFile)
        cFileOut = inFile.replace(".tif", "_standard.tif")
        standardFiles.append(cFileOut)
        if not os.path.exists(cFileOut):
            standardizeInputRasters(inR2, inR1, cFileOut)

    # Clip standard rasters to BALI, and then eliminate BALI
    baliFiles = []
    notBaliFiles = []
    for inFile in standardFiles:
        baliFile = inFile.replace(".tif", "_Bali.tif")
        baliFiles.append(baliFile)
        notBaliFile = inFile.replace(".tif", "_NotBali.tif")
        notBaliFiles.append(notBaliFile)
        logging.info(baliFile)
        if not os.path.exists(baliFile):
            curR = rasterio.open(inFile)
            if baliGeom.crs != curR.crs:
                baliGeom = baliGeom.to_crs(curR.crs)
            #Generate mask for Bali and Java
            baliGeoms = ((g, 1) for g in baliGeom['geometry'])
            baliMask = features.rasterize(baliGeoms, out_shape=curR.shape, transform=curR.transform)
            #Identify only Bali and Java
            baliImage = baliMask * curR.read()
            #Identify everythin else
            notBaliMask = np.abs(baliMask.astype("int8") - 1).astype("uint8")
            notBaliImage = notBaliMask * curR.read()
            #Write Bali Results
            with rasterio.open(baliFile, 'w', **curR.meta) as dst:
                dst.write(baliImage)
            #Write not Bali Results
            with rasterio.open(notBaliFile, 'w', **curR.meta) as dst:
                dst.write(notBaliImage)
    groupJaccard(outputFile, standardFiles)
    groupJaccard(outputFile.replace(".csv", "_Bali.csv"), baliFiles)
    groupJaccard(outputFile.replace(".csv", "_NotBali.csv"), notBaliFiles)

if __name__ == "__main__":
    exampleText = '''
    python rasterMisc.py -gdalbuildvrt -outFile Q:\GLOBAL\HYDRO\SSBN_Flooding\indonesia\ID-FD-5.vrt -file_list Q:\GLOBAL\HYDRO\SSBN_Flooding\indonesia\ID_fluvial_defended\ID-FD-5-1.tif Q:\GLOBAL\HYDRO\SSBN_Flooding\indonesia\ID_fluvial_defended\ID-FD-5-2.tif Q:\GLOBAL\HYDRO\SSBN_Flooding\indonesia\ID_fluvial_defended\ID-FD-5-3.tif Q:\GLOBAL\HYDRO\SSBN_Flooding\indonesia\ID_fluvial_defended\ID-FD-5-4.tif Q:\GLOBAL\HYDRO\SSBN_Flooding\indonesia\ID_fluvial_defended\ID-FD-5-5.tif Q:\GLOBAL\HYDRO\SSBN_Flooding\indonesia\ID_fluvial_defended\ID-FD-5-6.tif Q:\GLOBAL\HYDRO\SSBN_Flooding\indonesia\ID_fluvial_defended\ID-FD-5-7.tif Q:\GLOBAL\HYDRO\SSBN_Flooding\indonesia\ID_fluvial_defended\ID-FD-5-8.tif
    '''
    parser.add_argument("-gdalbuildvrt", dest="BUILDVRT", action="store_true", help="Build a VRT from input rasters")
    parser.add_argument("-file_list", dest="GDALFILES", nargs='+', action="store_true", help="List of files to build VRT from")
    parser.add_argument("-outFile", dest="VRTOUT", nargs='+', action="store_true", help="output vrt file")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
