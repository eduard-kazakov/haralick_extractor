
#coding: utf-8

import numpy as np
import mahotas
import PIL
import gdal


def get_haralick_texture_features_for_subimage (subimage):
    haralick = mahotas.features.haralick(subimage, True)
    if haralick.shape != (4, 13):
        haralick = np.zeros((4, 13)) + np.nan
    return haralick



def translate_array_to_grayscale (input_array, gray_levels, vmin=None, vmax=None):
    if not vmin:
        vmin = np.min(input_array)
    if not vmax:
        vmax = np.max(input_array)

    input_array = 1 + (gray_levels - 1)*(input_array.astype('float32') - vmin) / (vmax - vmin)
    
    input_array[input_array < 1] = 1
    input_array[input_array > gray_levels] = gray_levels

    return input_array.astype('uint8')


def calculate_haralick_features (subimages):
    harList = []

    for row in subimages:
        harRow = map(get_haralick_texture_features_for_subimage, row)
        harList.append(np.array(harRow))
    harImage = np.array(harList)

    harImageAnis = harImage.mean(axis=2)

    return np.swapaxes(harImageAnis.T, 1, 2)

def get_subimages_with_step_and_size (image_array, step, size):
    subimages = []
    for r in range(0, image_array.shape[0]-size+1, step):
        row_subimages = [image_array[r:r+size, c:c+size] for c in range(0, image_array.shape[1]-size+1, step)]
        subimages.append (row_subimages)
        
    return subimages


def recover_original_size_for_1_step_image (original_image_array, haralick_features_array):
    dif = len(original_image_array) - len(haralick_features_array[0])
    recovered_haralick_features_array =[]
    for haralick_feature in haralick_features_array:
        i = 0
        recovered_haralick = haralick_feature
        while i < dif/2:
            recovered_haralick = np.insert(recovered_haralick, 0, values=0, axis=1)
            recovered_haralick = np.insert(recovered_haralick, 0, values=0, axis=0)
            recovered_haralick = np.insert(recovered_haralick, len(recovered_haralick[0]), values=0, axis=1)
            recovered_haralick = np.insert(recovered_haralick, len(recovered_haralick), values=0, axis=0)
            
            i += 1
        
        recovered_haralick_features_array.append(recovered_haralick)

    return recovered_haralick_features_array
            

def save_haralick_as_images(haralick, path, recover_original_pixel_size=False, original_image_array = None, georeferencing=False, original_image_path=None):
    if recover_original_pixel_size:
        haralick_features = recover_original_size_for_1_step_image(original_image_array,haralick)
    else:
        haralick_features = haralick
        
    if georeferencing:
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
            outData.GetRasterBand(1).WriteArray(current_haralick)
            outData.SetProjection(projection)
            outData.SetGeoTransform(transform)
            del(outData)
            ind += 1
    
    else:
        ind = 0
        for current_haralick in haralick_features:
            current_haralick_image = PIL.Image.fromarray(current_haralick)
            current_path = path + "_haralick_" + str(ind) + ".tif"
            current_haralick_image.save(current_path)
            ind += 1


def create_haralick_features_for_geotiff (input_geotiff, output_dir_and_basename, step, size, grayscale_depth):
    processing_image = PIL.Image.open(input_geotiff)
    processing_image_array = np.asarray(processing_image)
    
    test_image_subimages = get_subimages_with_step_and_size(translate_array_to_grayscale(processing_image_array,grayscale_depth),step,size)
    
    haralick = calculate_haralick_features(test_image_subimages)
    
    if step == 1:
        save_haralick_as_images(haralick,output_dir_and_basename, True, processing_image_array, True, input_geotiff)
    else:
        save_haralick_as_images(haralick,output_dir_and_basename, False, None, False, None)


# USAGE EXAMPLE
# create_haralick_features_for_geotiff (<path to input raster>, <path to output path with base name>, <slicing window step>, <slicing window size>, <grayscale depth>)
create_haralick_features_for_geotiff ('C:/Users/ekazakov/TextureClassification/sample_data/sentinel_part/test2.tif','C:/Users/ekazakov/TextureClassification/sample_data/sentinel_part/test2_x10',1,10,128)
# It will create 13 geotiff files with names like test2_haralick_1, test2_haralick_2 etc. in folder C:/Users/ekazakov/TextureClassification/sample_data/sentinel_part/
