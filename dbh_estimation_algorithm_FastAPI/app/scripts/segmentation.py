# Commented out IPython magic to ensure Python compatibility.
import numpy as np
from PIL import Image
import PIL
import traceback
from app.scripts import pixel_analyzer as pa
from app.scripts import deeplab_model, runTilles
import os

domain = 'tree_trunk'

def getTreeDBH(filename, measured_dbh):
    '''
    Run some images as tiles if no tag was detected with running the full image
    '''
    try:
        # load image
        im = Image.open(filename)

        file = filename.split("/")[-1].split(".")[0]

        # run model 
        resized_img , seg_map = deeplab_model.dbh_MODEL.run(im)
        seg_image = deeplab_model.label_to_color_image(seg_map, domain).astype(np.uint8)

            # check if segmentation is not detected. 
        if not pa.isTagInMask(seg_image):
          print('Tag not detecte. Running tales')
          resized_img, seg_image = runTilles.runTilles(filename)

        # move to background later -- saved mask
        new_seg_iamge = Image.fromarray(np.uint8(seg_image)).convert('RGB')
        #new_seg_iamge.save('app/data/outputs/seg_image_original.png')
        #resized_img.save('app/data/outputs/resized_original_img.png')

        # save resized image for - segmentation evaluation
        #resized_img.save(f'app/data/outputs/resized_{file}.png')
        #new_seg_iamge.save(f'app/data/outputs/seg_image_{file}.png')

        # get tree-to-tag ratio and generate visualization
        generate_visualization = False
        dbh = pa.getTreePixelWidth(seg_image, file, measured_dbh, generate_visualization) 

        # if no tag is detected return the message
        if dbh == None:
          return "No tag detected"
  
        return dbh

    except Exception:
        print(traceback.format_exc())
        return "Execution failed"

def getTreeDBH2(filename, measured_dbh):
  try:
    # load image
    im = Image.open(filename)

    # filename
    file = filename.split("/")[-1].split(".")[0]

    # run model as tilled
    resized_img , seg_map = deeplab_model.dbh_MODEL.run(im)
    seg_image = deeplab_model.label_to_color_image(seg_map, domain).astype(np.uint8)

    # check orientation of tree in image - 1
    if not pa.isVertical(seg_image): # if tree is not vertical in the image flip the image 
      print(f"Image is rotated {filename}")
      im = im.transpose(Image.ROTATE_270)
      im.save(filename)
      resized_img , seg_map = deeplab_model.dbh_MODEL.run(im)
      seg_image = deeplab_model.label_to_color_image(seg_map, domain).astype(np.uint8)

    # get image dimension information
    width, height = im.size

    if width > height:
      print(f"Image is rotated {filename} with dimensions detection")
      im = im.transpose(Image.ROTATE_270)
      im.save(filename)
      resized_img , seg_map = deeplab_model.dbh_MODEL.run(im)
      seg_image = deeplab_model.label_to_color_image(seg_map, domain).astype(np.uint8)
      width, height = im.size

    ######## code for pixel percentage ######
    #file1 = "/Users/edwardamoah/Documents/GitHub/tree_dbh_estimation/data/static/tag_pixels.txt"
    #f = open(file1, "a")
    #f.write("\n")
    #f.write(file)
    #f.close()

    # check if segmentation is not detected. 
    if not pa.isTagInMask(seg_image):
      print('Tag not detecte. Running tales')
      resized_img, seg_image = runTilles.runTilles(filename)
   
    # save resized, and seg_imge for visualization
    new_seg_iamge = Image.fromarray(np.uint8(seg_image)).convert('RGB')
    #new_seg_iamge.save('app/data/outputs/seg_image_original_1.png')
    #resized_img.save('app/data/outputs/resized_original_img_1.png')

    # save original resized image - segmentation evaluation 
    #resized_img.save(f'app/data/outputs/resized_original_{file}.png')
    #new_seg_iamge.save(f'app/data/outputs/seg_image_original_{file}.png')

    # get zoom coordinates on tag
    left,top,right,bottom = pa.getZoomCordinates(seg_image, 100, resized_img, width, height)  

    # crop and save zooomed image
    zoomed_img = im.crop((left,top,right,bottom))
    zoomed_img.save(f'app/data/outputs/zoomed_img_{file}.png') 

    # get dbh on zoomed image
    dbh = getTreeDBH(f'app/data/outputs/zoomed_img_{file}.png', measured_dbh)

    # remove zoomed image
    os.remove(f'app/data/outputs/zoomed_img_{file}.png')

    return dbh
  except:
    print(traceback.format_exc())
    return None