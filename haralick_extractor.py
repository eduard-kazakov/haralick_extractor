
# coding: utf-8

# In[1]:

import numpy as np
import mahotas
import PIL
import gdal


# In[2]:

def get_haralick_texture_features_for_subimage (subimage):
    #print subimage
    try:
        haralick = mahotas.features.haralick(subimage, True)
        if haralick.shape != (4, 13):
            haralick = np.zeros((4, 13)) + np.nan
        return haralick
    except:
        haralick = np.zeros((4, 13)) + np.nan
        return haralick

# In[3]:

def translate_array_to_grayscale (input_array, gray_levels, vmin=None, vmax=None):
    
    if not vmin:
        vmin = np.nanmin(input_array)
    if not vmax:
        vmax = np.nanmax(input_array)

    input_array = 1 + (gray_levels - 1)*(input_array.astype('float32') - vmin) / (vmax - vmin)
    
    input_array[input_array < 1] = 1
    input_array[input_array > gray_levels] = gray_levels

    # return as unsigned integer
    return input_array.astype('uint8')


# In[4]:

def calculate_haralick_features (subimages):
    # init parallel processing
    #pool = Pool(threads)

    # apply calculation of Haralick texture features in many threads
    # in row-wise order
    harList = []

    for row in subimages:
        #print row[0]
        harRow = map(get_haralick_texture_features_for_subimage, row)
        #print '---'
        #print harRow
        harList.append(np.array(harRow))
        #if np.array(harRow).shape != (len(subImgs), 4, 13):
        #    raise

    # convert list with texture features to array
    harImage = np.array(harList)

    # calculate directional mean
    harImageAnis = harImage.mean(axis=2)

    #pool.close()
    # reshape matrix and make images to be on the first dimension
    return np.swapaxes(harImageAnis.T, 1, 2)


# In[5]:

def get_subimages_with_step_and_size (image_array, step, size):
    
    subimages = []
    for r in range(0, image_array.shape[0]-size+1, step):
        row_subimages = [image_array[r:r+size, c:c+size] for c in range(0, image_array.shape[1]-size+1, step)]
        subimages.append (row_subimages)
        
    return subimages


# In[10]:

def recover_original_size_for_1_step_image (original_image_array, haralick_features_array):
    dif = len(original_image_array) - len(haralick_features_array[0])
    recovered_haralick_features_array =[]
    for haralick_feature in haralick_features_array:
        i = 0
        recovered_haralick = haralick_feature
        while i < dif/2:
            recovered_haralick = np.insert(recovered_haralick, 0, values=np.nan, axis=1)
            recovered_haralick = np.insert(recovered_haralick, 0, values=np.nan, axis=0)
            recovered_haralick = np.insert(recovered_haralick, len(recovered_haralick[0]), values=np.nan, axis=1)
            recovered_haralick = np.insert(recovered_haralick, len(recovered_haralick), values=np.nan, axis=0)
            
            i += 1
        
        recovered_haralick_features_array.append(recovered_haralick)

    return recovered_haralick_features_array
            
    


# In[7]:

def save_haralick_as_images(haralick, path, recover_original_pixel_size=False, original_image_array = None, original_georeferencing=False, deformed_georeferencing=False, dif_x = None, dif_y = None, step = None, original_image_path=None):
    if recover_original_pixel_size:
        haralick_features = recover_original_size_for_1_step_image(original_image_array,haralick)
    else:
        haralick_features = haralick
        
    if original_georeferencing:
        original_raster = gdal.Open(original_image_path)
        projection = original_raster.GetProjection()
        transform = original_raster.GetGeoTransform()
        bands = 1
        cols = original_raster.RasterXSize
        rows =  original_raster.RasterYSize
        dt = gdal.GDT_Float32
        format = 'GTiff'
        driver = gdal.GetDriverByName(format)
        
        ind = 0
        for current_haralick in haralick_features:
            print cols, rows
            outData = driver.Create(path+'_haralick_' + str(ind) + '.tif', cols, rows, bands, dt)
            #print np.max(current_haralick)
            outData.GetRasterBand(1).WriteArray(current_haralick)
            outData.SetProjection(projection)
            outData.SetGeoTransform(transform)
            del(outData)
            ind += 1
    
    elif deformed_georeferencing:
        original_raster = gdal.Open(original_image_path)
        projection = original_raster.GetProjection()
        transform = original_raster.GetGeoTransform()
        bands = 1
        
        cols = len(haralick_features[0][0])
        rows =  len(haralick_features[0])
        
        dt = gdal.GDT_Float32
        format = 'GTiff'
        driver = gdal.GetDriverByName(format)
        
        ind = 0
        for current_haralick in haralick_features:
            print cols, rows
            outData = driver.Create(path+'_haralick_' + str(ind) + '.tif', cols, rows, bands, dt)
            #print np.max(current_haralick)
            outData.GetRasterBand(1).WriteArray(current_haralick)
            outData.SetProjection(projection)
            outData.SetGeoTransform([transform[0],transform[1]*step,transform[2],transform[3],transform[4],transform[5]*step])
            del(outData)
            ind += 1
        
    else:
        ind = 0
        for current_haralick in haralick_features:
            current_haralick_image = PIL.Image.fromarray(current_haralick)
            current_path = path + "_haralick_" + str(ind) + ".tif"
            current_haralick_image.save(current_path)
            ind += 1


# In[8]:

def create_haralick_features_for_geotiff (input_geotiff, output_dir_and_basename, step, size, grayscale_depth):
    processing_image = PIL.Image.open(input_geotiff)
    processing_image_array = np.asarray(processing_image)
    
    image_subimages = get_subimages_with_step_and_size(translate_array_to_grayscale(processing_image_array,grayscale_depth),step,size)
    
    haralick = calculate_haralick_features(image_subimages)
    
    if step == 1:
        save_haralick_as_images(haralick = haralick,
                                path = output_dir_and_basename,
                                recover_original_pixel_size = True,
                                original_image_array = processing_image_array,
                                original_georeferencing = True,
                                deformed_georeferencing = False,
                                dif_x = None,
                                dif_y = None,
                                step = None,
                                original_image_path = input_geotiff)
    elif step == size:
        # 227 x 108
        #print dif_x, dif_y
        dif_y = len(processing_image_array) - len(image_subimages)*size
        dif_x = len(processing_image_array[0]) - len(image_subimages[0])*size
        save_haralick_as_images(haralick = haralick,
                                path = output_dir_and_basename,
                                recover_original_pixel_size = False,
                                original_image_array = processing_image_array,
                                original_georeferencing = False,
                                deformed_georeferencing = True,
                                dif_x = dif_x,
                                dif_y = dif_y,
                                step = step,
                                original_image_path = input_geotiff)
    else:
        save_haralick_as_images(haralick = haralick,
                                path = output_dir_and_basename,
                                recover_original_pixel_size = False,
                                original_image_array = None,
                                original_georeferencing = False,
                                deformed_georeferencing = False,
                                dif_x = None,
                                dif_y = None,
                                step = None,
                                original_image_path = False)
        

# In[11]:

# USAGE EXAMPLE
# create_haralick_features_for_geotiff (<path to input raster>, <path to output path with base name>, <slicing window step>, <slicing window size>, <grayscale depth>)
create_haralick_features_for_geotiff ('E:/LENA/SPb/SPb_29_12_16/park.tif','E:/LENA/SPb/SPb_29_12_16/park_10_',10,10,256)
# It will create 13 geotiff files with names like test2_haralick_1, test2_haralick_2 etc. in folder C:/Users/ekazakov/TextureClassification/sample_data/sentinel_part/

