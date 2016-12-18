
# coding: utf-8

# In[56]:

import numpy as np
import mahotas
import PIL
import gdal


# In[66]:

def get_noise_metrics_for_subimage (subimage):
    
    mean = np.mean(subimage)
    stdev = np.std(subimage)
    noise = stdev/mean
    
    return [mean,stdev,noise]


# In[ ]:




# In[67]:

def translate_array_to_grayscale (input_array, gray_levels, vmin=None, vmax=None):
    if not vmin:
        vmin = np.min(input_array)
    if not vmax:
        vmax = np.max(input_array)

    input_array = 1 + (gray_levels - 1)*(input_array.astype('float32') - vmin) / (vmax - vmin)
    
    input_array[input_array < 1] = 1
    input_array[input_array > gray_levels] = gray_levels

    # return as unsigned integer
    return input_array.astype('uint8')


# In[68]:

def calculate_noise_metrics (subimages):
    # init parallel processing
    #pool = Pool(threads)

    # in row-wise order
    noiseList = []

    for row in subimages:
        #print row[0]
        noiseRow = map(get_noise_metrics_for_subimage, row)
        #print '---'
        #print harRow
        noiseList.append(np.array(noiseRow))
        #if np.array(harRow).shape != (len(subImgs), 4, 13):
        #    raise

    # convert list with texture features to array
    noiseImage = np.array(noiseList)
    #print noiseImage
    # calculate directional mean
    noiseImageAnis = noiseImage

    #pool.close()
    # reshape matrix and make images to be on the first dimension
    #return noiseImageAnis
    return np.swapaxes(noiseImageAnis.T, 1, 2)


# In[69]:

def get_subimages_with_step_and_size (image_array, step, size):
    subimages = []
    for r in range(0, image_array.shape[0]-size+1, step):
        row_subimages = [image_array[r:r+size, c:c+size] for c in range(0, image_array.shape[1]-size+1, step)]
        subimages.append (row_subimages)
        
    return subimages


# In[ ]:




# In[70]:

def recover_original_size_for_1_step_image (original_image_array, noise_metrics_array):
    dif = len(original_image_array) - len(noise_metrics_array[0])
    recovered_noise_metrics_array =[]
    for noise_metrics in noise_metrics_array:
        i = 0
        recovered_noise_metrics = noise_metrics
        while i < dif/2:
            recovered_noise_metrics = np.insert(recovered_noise_metrics, 0, values=np.nan, axis=1)
            recovered_noise_metrics = np.insert(recovered_noise_metrics, 0, values=np.nan, axis=0)
            recovered_noise_metrics = np.insert(recovered_noise_metrics, len(recovered_noise_metrics[0]), values=np.nan, axis=1)
            recovered_noise_metrics = np.insert(recovered_noise_metrics, len(recovered_noise_metrics), values=np.nan, axis=0)
            
            i += 1
        
        recovered_noise_metrics_array.append(recovered_noise_metrics)

    return recovered_noise_metrics_array
            
    


# In[ ]:




# In[74]:

def save_noise_metrics_as_images (noise_metrics, path, recover_original_pixel_size=False, original_image_array = None, original_georeferencing=False, deformed_georeferencing=False, dif_x = None, dif_y = None, step = None, original_image_path=None):
    if recover_original_pixel_size:
        noise_metrics_features = recover_original_size_for_1_step_image(original_image_array,noise_metrics)
    else:
        noise_metrics_features = noise_metrics
        
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
        for current_noise_metrics in noise_metrics_features:
            print cols, rows
            outData = driver.Create(path+'_noise_metrics_' + str(ind) + '.tif', cols, rows, bands, dt)
            #print np.max(current_haralick)
            outData.GetRasterBand(1).WriteArray(current_noise_metrics)
            outData.SetProjection(projection)
            outData.SetGeoTransform(transform)
            del(outData)
            ind += 1
    
    elif deformed_georeferencing:
        original_raster = gdal.Open(original_image_path)
        projection = original_raster.GetProjection()
        transform = original_raster.GetGeoTransform()
        bands = 1
        
        cols = len(noise_metrics_features[0][0])
        rows =  len(noise_metrics_features[0])
        
        dt = gdal.GDT_Float32
        format = 'GTiff'
        driver = gdal.GetDriverByName(format)
        
        ind = 0
        for current_noise_metrics in noise_metrics_features:
            print cols, rows
            outData = driver.Create(path+'_noise_metrics_' + str(ind) + '.tif', cols, rows, bands, dt)
            outData.GetRasterBand(1).WriteArray(current_noise_metrics)
            outData.SetProjection(projection)
            outData.SetGeoTransform([transform[0],transform[1]*step,transform[2],transform[3],transform[4],transform[5]*step])
            del(outData)
            ind += 1
        
    else:
        ind = 0
        for current_noise_metrics in noise_metrics_features:
            current_noise_metrics_image = PIL.Image.fromarray(current_noise_metrics)
            current_path = path + "_noise_metrics_" + str(ind) + ".tif"
            current_noise_metrics_image.save(current_path)
            ind += 1


# In[8]:

def create_noise_metrics_features_for_geotiff (input_geotiff, output_dir_and_basename, step, size):
    processing_image = PIL.Image.open(input_geotiff)
    processing_image_array = np.asarray(processing_image)
    
    image_subimages = get_subimages_with_step_and_size(processing_image_array,step,size)
    
    noise_metrics = calculate_noise_metrics(image_subimages)
    
    if step == 1:
        save_noise_metrics_as_images(noise_metrics= noise_metrics,
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
        save_noise_metrics_as_images(noise_metrics = noise_metrics,
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
        save_noise_metrics_as_images(noise_metrics = noise_metrics,
                                path = output_dir_and_basename,
                                recover_original_pixel_size = False,
                                original_image_array = None,
                                original_georeferencing = False,
                                deformed_georeferencing = False,
                                dif_x = None,
                                dif_y = None,
                                step = None,
                                original_image_path = False)
        

# In[76]:

# USAGE EXAMPLE
# create_noise_metrics_features_for_geotiff (<path to input raster>, <path to output path with base name>, <slicing window step>, <slicing window size>)
create_noise_metrics_features_for_geotiff ('E:/LENA/SPb/SPb_29_12_16/park.tif','E:/LENA/SPb/SPb_29_12_16/park_10_',10,10)
# It will create 3 geotiff files with names like test2_noise_metrics_1, test2_noise_metrics_2 etc. in folder C:/Users/ekazakov/TextureClassification/sample_data/sentinel_part/
# 0 is mean; 1 is stdev; 2 is speckle noise
