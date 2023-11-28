from image_slicer import Tile, slice as slice_image
from scripts import deeplab_model 
from PIL import Image
import numpy as np
from numpy import asarray
import os

# load image
def runTile(filename, tile_number):
    im = Image.open(filename)

    # run model 
    resized_img , seg_map = deeplab_model.dbh_MODEL.run(im)
    seg_image = deeplab_model.label_to_color_image(seg_map, deeplab_model.domain).astype(np.uint8)

    # move to background later -- saved mask
    new_seg_iamge = Image.fromarray(np.uint8(seg_image)).convert('RGB')
    new_seg_iamge.save(f'app/data/outputs/temp_tile_{tile_number}.png')
    resized_img.save(f'app/data/outputs/resized_img_{tile_number}.png')

    return [f'app/data/outputs/temp_tile_{tile_number}.png', f'app/data/outputs/resized_img_{tile_number}.png']


def stitch_tiles_together(tile_paths) -> Image.Image:
        """Stitch the tiles back together."""

        # get the dimensions of each tile
        with Image.open(tile_paths[0]) as mask_tile_image:
            mask_tile_width = mask_tile_image.width
            mask_tile_height = mask_tile_image.height

        # new full-size segmentation mask with the same dimensions as original image
        image = Image.new('RGB', (mask_tile_width*3, mask_tile_height*3), (0, 0, 0))

        # Combine segmentation mask tiles back together
        row = 0
        col = 0
        for p in tile_paths:
            with Image.open(p) as image_tile:
                #print(image_tile.size)

                #detected_class_indices: List[int] = np.unique(image).tolist()
                #print(f"tile: {p} has classes: {detected_class_indices}")

                # determine where in the full segmentation mask the tile should go
                offset = mask_tile_width * col, mask_tile_height * row

                image.paste(image_tile, offset)

                # update row/col value based on where previous tile was pasted
                if col != 2:
                    col += 1
                else:
                    row += 1
                    col = 0

        return image


def runTilles(path):
    # tile the image
    tiles= slice_image(path, 9)

    # run model on tiles
    mask_paths = []
    resized_img_paths = []
    for i in range(len(tiles)):
        mask_file, img_file = runTile(tiles[i].filename, i)
        mask_paths.append(mask_file)
        resized_img_paths.append(img_file)

    # put mask together
    seg_image = stitch_tiles_together(mask_paths)

    # stich resized images together
    resized_img = stitch_tiles_together(resized_img_paths)
    resized_img.save('app/data/outputs/resized_img.png')

    # convert image into numpy arrray
    seg_image = asarray(seg_image)

    # remove image and mask tiles
    for mask in mask_paths:
        try:
            os.remove(mask)
        except:
            continue
    
    for tile in tiles:
        try:
            os.remove(tile.filename)
        except:
            continue

    # return seg_image
    return resized_img, seg_image