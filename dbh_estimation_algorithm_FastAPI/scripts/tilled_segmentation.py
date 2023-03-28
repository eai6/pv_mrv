from dataclasses import dataclass
from enum import unique
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Set, Tuple

from image_slicer import Tile, slice as slice_image # type: ignore
#from lambda_multiprocessing import Pool
import numpy as np # type: ignore
from matplotlib import pyplot as plt # type: ignore
from PIL import Image

from deeplab_model import DeepLabModel

@dataclass(frozen=True)
class MaskFilePaths:
    """Struct class for holding mask image paths."""
    segmentation_mask_path: Path
    color_mask_path: Path
    image_overlay_mask_path: Path


class WheatRustProcessor():
    """Utility class for running wheat rust semantic segmentation models.
    Public attrbutes:
    mask_paths: A MaskFilePaths object
    detected_classes: A set of strings containing the detected classes.
    """

    # label colormap used in PASCAL VOC segmentation benchmark.
    _label_colormap = np.array([
        [0,     0,   0],
        [255, 255, 255],
        [0,    85,   0],
        [170,   0,   0],
        [255, 255,   0],
        [208, 139,   0],
        [100, 255,   0]
    ])

    # Label colormap for the wheat stem rust dataset.
    _label_maskmap = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
        [6, 6, 6]
    ])

    # The number of tiles to split images into.
    _num_tiles = 9

    class_names = [
        "unlabeled",
        "background",
        "Wheat Leaf",
        "Wheat Stem Rust",
        "Wheat Stripe Rust",
        "Wheat Leaf Rust",
        "Wheat Stem"
    ]

    def __init__(self, input_image_path: Path, output_mask_paths: MaskFilePaths,
                modelpath: Path, num_processes: int, percent_threshold: float):
        """
        Args:
        input_image_path: A Path object to the input image.
        output_mask_paths: A MaskFilePaths object.
        model: A SemanticSegmentation model to be used on the image tiles.
        num_processes: The number of worker processes to spawn. Must be in range [1, 9]
        percent_threshold: The percent of pixels a class must have to be included in the result. Must be in range [0, 1]
        """
        self._input_image_path = input_image_path
        self.mask_paths = output_mask_paths
        self._modelpath = modelpath
        self._num_processes = num_processes
        self._percent_threshold = percent_threshold

        if self._num_processes < 1:
            print(f"num_processes < 1: {self._num_processes}. Setting to 1.")
            self._num_processes = 1

        if self._num_processes > 9:
            print(f"num_processes > 9: {self._num_processes}. Setting to 9.")
            self._num_processes = 9

        if self._percent_threshold < 0:
            print(f"percent_threshold < 0: {self._percent_threshold}. Setting to 0")
            self._percent_threshold = 0

        if self._percent_threshold > 1:
            print(f"percent_threshold > 1: {self._percent_threshold}. Setting to 1.0")
            self._percent_threshold = 1

        class_label_length = len(np.asarray(self.class_names))
        full_label_map = np.arange(class_label_length).reshape(class_label_length, 1)

        self.full_color_map = \
            WheatRustProcessor._label_to_color_image(full_label_map)


    @staticmethod
    def _label_to_color_image(label: np.ndarray) -> np.ndarray:
        """Adds color defined by the dataset colormap to the label.
        Args:
        label: A 2D array with integer type, storing the segmentation label.
        Returns:
        result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.
        Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
        """
        if label.ndim != 2:
            raise ValueError('Expect 2-D input label')

        if np.max(label) >= len(WheatRustProcessor._label_colormap):
            raise ValueError('label value too large.')

        return WheatRustProcessor._label_colormap[label]


    @staticmethod
    def _label_to_mask_image(label: np.ndarray) -> np.ndarray:
        """Doesn't really do anything...?
        but the original masks aren't showing up like I expected
        (they have colors when they should be black and white)
        Args:
        label: A 2D array with integer type, storing the segmentation label.
        Returns:
        result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.
        Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
        """
        if label.ndim != 2:
            raise ValueError('Expect 2-D input label')

        if np.max(label) >= len(WheatRustProcessor._label_maskmap):
            raise ValueError('label value too large.')

        return WheatRustProcessor._label_maskmap[label]


    @staticmethod
    def _save_image(image: Image.Image, output_path: Path) -> None:
        """Helper to save an image to a file with consistent formatting."""
        plt.imshow(image, alpha=1.0)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


    def _process_tile(self, tile: Tile, output_dir: Path) -> Tuple[int, Path, Path]:
        """Process an individual tile."""

        tile_path = Path(tile.filename)
        suffix = self._input_image_path.suffix

        mask_overlay_filename = Path(tile_path.stem + f"_mask_{tile.number}")
        mask_overlay_filename = mask_overlay_filename.with_suffix(suffix)

        color_mask_overlay_filename = Path(tile_path.stem + f"_color_mask_{tile.number}")
        color_mask_overlay_filename = color_mask_overlay_filename.with_suffix(suffix)

        # define the full output paths for the mask and color mask tiles
        output_tile_mask_path = output_dir / mask_overlay_filename
        output_tile_color_path = output_dir / color_mask_overlay_filename

        print(f"in pid {os.getpid()}: output_tile_mask_path = {output_tile_mask_path} ; output_tile_color_path  {output_tile_color_path}")

        model = DeepLabModel(self._modelpath)
        with Image.open(tile.filename) as img_tile:
            print(f"in pid {os.getpid()} - running model.")
            _, seg_tile_map = model.run(img_tile)

        seg_tile_color_image = \
            WheatRustProcessor._label_to_color_image(seg_tile_map).astype(np.uint8)
        seg_tile_mask_image = \
            WheatRustProcessor._label_to_mask_image(seg_tile_map).astype(np.uint8)

        # save the masks to files
        WheatRustProcessor._save_image(seg_tile_color_image, output_tile_color_path)
        WheatRustProcessor._save_image(seg_tile_mask_image, output_tile_mask_path)

        return (tile.number, output_tile_mask_path, output_tile_color_path)


    def _stitch_tiles_together(self, tile_paths: List[Path]) -> Image.Image:
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

                detected_class_indices: List[int] = np.unique(image).tolist()
                print(f"tile: {p} has classes: {detected_class_indices}")

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


    def _get_detected_classes(self, image: Image) -> List[int]:
        """Print the classes and portions in the full image."""
        image_array = np.array(image)
        print(f"input image has shape: {image_array.shape}")

        detected_class_indices: List[int] = np.unique(image).tolist()

        # remove spurious 255 values that the model adds.
        if 255 in detected_class_indices:
            detected_class_indices.remove(255)

        num_pixels = image.height * image.width * 1.0

        print(f"detected class indices: {detected_class_indices}")
        class_names = [self.class_names[c] for c in detected_class_indices]
        print(f"unique values: {detected_class_indices}: {class_names}")
        for detected_class in detected_class_indices:
            num_class_pixels = np.count_nonzero(
                np.all(image_array == detected_class, axis=2)
            )
            class_name = self.class_names[detected_class]

            percent_pixels = round(num_class_pixels / num_pixels, 4)
            if percent_pixels < self._percent_threshold:
                print(f"Removing class {class_name} because its percent {percent_pixels} is less than the cutoff of {self._percent_threshold}.")
                detected_class_indices.remove(detected_class)

        return detected_class_indices


    def _split_and_recombine_tiles(self) -> None:
        """
        Slices the image into self._num_tiles tiles, runs each tile through the model,
        recombines the tiles into a full size mask and color mask,
        saves the full masks and cleans up the tiles
        """

        tiles: Tuple[Tile] = slice_image(self._input_image_path.as_posix(), self._num_tiles)
        print('Running Tiles Through Deeplab')

        with TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            print(f"using {self._num_processes} processes")
            with Pool(self._num_processes) as pool:
                path_list = [tmp_dir_path for t in tiles]
                outputs = pool.starmap(self._process_tile, zip(tiles, path_list))

            segmentation_mask_tile_filepaths = [o[1] for o in outputs]
            color_mask_tile_filepaths = [o[2] for o in outputs]

            self.segmentation_mask = self._stitch_tiles_together(segmentation_mask_tile_filepaths)
            self.color_mask = self._stitch_tiles_together(color_mask_tile_filepaths)

            self.detected_classes = self._get_detected_classes(self.segmentation_mask)
            print(f'Classes recognized in this full seg mask: {self.detected_classes}')

            # save the output mask images
            self.segmentation_mask.save(self.mask_paths.segmentation_mask_path)
            self.color_mask.save(self.mask_paths.color_mask_path)

            # XXX: Do we need to close self.segmentation_mask and self.color_mask?
            self.segmentation_mask.close()
            self.color_mask.close()

        print('visualizing full image and prediction mask..')
        self._vis_segmentation_overlay()


    def _vis_segmentation_overlay(self)-> None:
        """Visualizes segmentation map overlaid on top of input image"""

        # open the segmentation mask and the image
        with Image.open(self.mask_paths.color_mask_path) as color_mask, \
                Image.open(self._input_image_path) as image:

            # resize mask to fit dimensions of image
            image = image.resize((color_mask.width, color_mask.height))

            # display the image
            plt.imshow(image)
            plt.axis('off')

            # display the color mask over the image with alpha - 0.7
            plt.imshow(color_mask, alpha=0.7)

            # save the image with segmentation mask overlaid
            plt.savefig(self.mask_paths.image_overlay_mask_path, dpi=400,
                        bbox_inches='tight', pad_inches=0)


    def run_visualization(self) -> Tuple[List[int], List[str]]:
        """Inferences DeepLab model and visualizes result."""
        print('Splitting images into tiles...')
        self._split_and_recombine_tiles()

        class_labels = [self.class_names[c] for c in self.detected_classes]

        return (self.detected_classes, class_labels)