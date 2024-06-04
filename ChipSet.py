#!/usr/bin/env python
'''The GeoChipSet module contains classes for chipping large-format geospatial images that
are typically collected from satellites, airplanes, or drones into small, manageable "chips"
for use in computer vision applications.

TODO (NOW):
- Convert functions to make use of yield generator where neccecary.
- Possibly consolidate transform generation code into stack_chips()?
- Unit tests for everything, but especially the unstack_chips(), mask_ndv() methods.
- Complete documentation w/ readme.
- Verbose Param/Logging: ...
- Implement basic GeoInterface and other expected geospatial software stuff... (could prob just use rio/fio?).

TODO (SOMEDAY):
- OVERLAP (REVERSABLE OPS): Add ability to do non-reversable operations (transforms) w/ some type of acceleration.
  Ideas:
    1. Dask, allow user to make most of their machine
    2. PyTorch's unfold/fold operations. Send to GPU.
    3. Old and slow approach, just warn the user and provide better options if speed is critical.
- NON-GEO/EXIF CHIPSETS: This operates on aerial photos w/o pixel-level georeferencing info,
  potentially making use of image EXIF data.
- Include label measurement tools: since we know so much about GeoChipSets we could have a function
  that auto-appends columns with label measures in real life (i.e. sq. meters, sq. feet, etc.)
- SuperChipSet: wrap the ChipSet classes in parent classes that allow operations on sets of
  ChipSets (i.e. an entire folder of images, each is its own GeoChipSet w/ unique attributes.)
    - IN-MEM vs ON-DISK: If we move to SuperChipSets we will need the ability to save as HDF5 and/or GeoZarr/Zarr
      files on disk, access in chunks.
- Prettier plots: potential for interactive maps / plots. Or at least maps w/ coordinate info and
  basemaps to ensure things are lining up appropriately.
- Explore embedding image NDV masks into the arrays themselves (np.masked_array) instead of storing
  as separate sidecar files.
- Replace pandas/geopandas with shapely/fiona to increase efficiency and reduce dependency requirements.
- Custom formats (raster and vector).
'''

import os
import copy
import math
import random
import itertools

import numpy as np
import geopandas as gpd
import shapely.geometry as geom
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.errors import RasterioIOError
from rasterio.plot import reshape_as_image, reshape_as_raster


class GeoChipSet(object):
    def __init__(self, src_path: str):
        # Class Constants
        self.GEOCHIPSET_DTYPE = np.dtype([('chip', object),
                                          ('chip_mask', object),
                                          ('topleft_anchor', object),
                                          ('is_invalid', object),
                                          ('transform', object),
                                          ('geom', object)])

        # Attributes created/modified by __init__()...
        self.src_path = src_path  # the path to the source (input) image on disk.
        self.array = np.empty((0, 0, 0), dtype=np.float64)  # the array extracted from the source image.
        self.image = np.empty((0, 0, 0), dtype=np.float64)  # the array reshaped to standard image axis ordering (PIL).
        self.src_nodata: int = None  # the source image's NoData Value, which denotes pixels that contain no valid data.
        self.src_profile = None  # the source image's Rasterio profile.

        # Attributes created/modified by .stack_chips()...
        self.chipset = None

        # Attributes created/modified by .index_chipset()...
        self.chipset_index: gpd.GeoDataFrame = None  # a GeoPandas GeoDataFrame built from chip_dict.

        # Open the image file, read in any data and all available geospatial metadata using Rasterio.
        try:
            self.src_name, self.src_ext = os.path.splitext(src_path)

            with rio.open(self.src_path) as src:  # TODO: consolidate these attrs w/ the above, figure out exactly what we need to keep.
                self.src_shape = (src.height, src.width, src.count)
                self.src_bounds = src.bounds
                self.src_crs = src.crs  # TODO: GeoChipSet should enforce CRS.
                self.src_transform = src.transform
                self.src_nodata = src.nodata
                self.src_dtype = src.dtypes[0]
                self.src_colorinterp = src.colorinterp
                self.src_profile = src.profile

                self.array = src.read()
                self.array_img = reshape_as_image(self.array)
        except RasterioIOError as e:
            print(f"RasterioIOError: {e}. An object was created, but it does not contain any valid image data.")  # TODO: This is a better format.
        except Exception as e:
            print(f"An unexpected error occured when opening image with Rasterio: {e}.")

    def show_src_image(self) -> None:
        '''Show a simple plot of the source image using matplotlib.imshow().'''
        if self.array_img is None:
            raise ValueError("A valid array was not extracted from the source image. Please check your input image.")

        try:
            plt.imshow(self.array_img)
            plt.show()
        except Exception as e:
            print(f"An unexpected error occured when showing source image: {e}.")
            raise

    def show_chip(self, idx: int = 0) -> None:
        '''Show a simple plot of a chip at position idx using matplotlib.imshow().'''
        if self.chipset is None:
            raise ValueError("No ChipSet is present, run self.stack_chips() first.")

        try:
            plt.imshow(self.chipset[idx]['chip'])
            plt.show()
        except Exception as e:
            print(f"An unexpected error occured when showing image chip (index: {idx}): {e}")
            raise

    def yield_chips(self):
        '''Utility method that returns a generator for the ChipSet.'''
        if self.chipset is None:
            raise ValueError("No ChipSet is present, run .stack_chips() first.")

        try:
            yield from self.chipset
        except Exception as e:
            print(f"An unexpected error occured while yielding ChipSet: {e}.")

    def show_random_chip(self, masked: bool = True) -> None:
        '''Show a simple plot of a chip at a random position using matplotlib.imshow().'''
        if self.chipset is None:
            raise ValueError("No ChipSet is present, run .stack_chips() first.")

        try:
            if masked is True:
                choices = np.where(~self.chipset['is_invalid'])[0].tolist()
            else:
                choices = list(range(len(self.chipset)))

            idx = random.choice(choices)
            self.show_chip(idx)
        except Exception as e:
            print(f"An unexpected error occured when showing a random image chip (index: {idx}): {e}")
            raise

    def index_chipset(self) -> None:
        '''Computes per-chip statistics such as min/max pixels, geometries, and affine
        transformation matricies. This data is then stored as both a dictionary and a
        GeoPandas GeoDataFrame() object.

        The geometries will be square or rectangular polygons which represent the extent of each
        chip in the ChipSet. All geometries will be stored in the source image's CRS. The chip
        index can be useful for visualizing the bounding boxes of each chip on a map.

        # TODO
        Args:
            None

        Returns:
            None

        Depends:
            self.chipset (np.Array): a numpy array of shape (num_chips, chip_height, chip_width, chip_bands)
                that contains each chip created by .stack_chips().

        Modifies:
            self.chipset_index (gpd.GeoDataFrame): a class attribute that stores the final chip
                index file as a GeoPandas GeoDataFrame object.
        '''
        if self.chipset is None:
            raise ValueError("No ChipSet is present, run .stack_chips() first.")

        try:
            # begin assembling the GeoDataFrame on a column-by-column basis
            idx_col = [x for x in range(1, (self.num_chips + 1))]  # index col
            src_col = [str(self.src_path)] * self.num_chips  # source image path col

            chip_h, chip_w = self.chip_hw
            tls = self.chipset['topleft_anchor'].tolist()
            brs = [[tl[0] + chip_h, tl[1] + chip_w] for tl in tls]
            y_min_col = [tl[0] for tl in tls]  # minimum y col
            x_min_col = [tl[1] for tl in tls]  # minimum x col
            y_max_col = [br[0] for br in brs]  # maximum y col
            x_max_col = [br[1] for br in brs]  # maximum x col

            # geom_col = []  # chip geometry col
            with rio.open(self.src_path, 'r') as src:  # TODO: transform should move to stack_chips?
                for i, chip in enumerate(self.yield_chips()):
                    win = rio.windows.Window(chip['topleft_anchor'][1], chip['topleft_anchor'][0],
                                             chip_h, chip_w)
                    win_transform = rio.windows.transform(win, src.transform)
                    win_bounds = rio.windows.bounds(win, src.transform)

                    chip['geom'] = geom.box(*win_bounds, ccw=False)
                    # geom_col.append(geom.box(*win_bounds, ccw=False))
                    chip['transform'] = win_transform

                chipset_attributes = {'order': idx_col, 'source': src_col,
                                      'geometry': self.chipset['geom'],
                                      'y_min': y_min_col, 'x_min': x_min_col,
                                      'y_max': y_max_col, 'x_max': x_max_col}
                self.chipset_index = gpd.GeoDataFrame(chipset_attributes, crs=self.src_crs)
                print("self.chipset_index created.")
        except Exception as e:
            print(f"An unexpected error occured when indexing the chipset: {e}")
            raise

    def show_chipset_index(self) -> None:
        '''Show a simple plot of the chipset index on a map using geopandas.plot().'''
        if self.chipset_index is None:
            raise ValueError("No Chip Index is present, run .index_chipset() first.")

        try:
            self.chipset_index.plot()
            plt.show()
        except Exception as e:
            print(f"An unexpected error occured when plotting the Chip Index: {e}")

    def save_chipset_index(self, out_dir: str) -> None:
        '''Save a Chip Index to disk.'''
        if self.chipset_index is None:
            raise ValueError("No Chip Index is present, run .index_chipset() first.")

        try:
            self.chipset_index.to_file(out_dir, index=True)
        except Exception as e:
            print(f"An unexpected error occured when saving the chip index to disk: {e}.")

    def save_chips(self, out_dir: str) -> None:
        '''Loop over self.chipset and write each image to disk with rasterio.'''
        if self.chipset is None:
            raise ValueError("No ChipSet is present, run self.stack_chips() first.")

        try:
            print("Writing chips to disk...")
            basename, ext = os.path.splitext(os.path.basename(self.src_path))
            for i, chip in enumerate(self.yield_chips()):
                print(f"Writing chip {i}.")

                chip_h, chip_w = self.chip_hw
                profile = copy.deepcopy(self.src_profile)
                profile.update({'transform': chip['transform'],
                                'nodata': self.src_nodata,
                                'height': chip_h,
                                'width': chip_w})
                row, col = chip['topleft_anchor']
                filepath = os.path.join(out_dir, f"{basename}_{row}_{col}{ext}")
                print(filepath)
                with rio.open(filepath, 'w', **profile) as dst:
                    dst.write(reshape_as_raster(chip['chip']))
        except Exception as e:
            print(f"An unexpected error occured when saving the chip images to disk: {e}.")

    @staticmethod
    def _mask_ndvs(arr: np.ndarray, ndv: int) -> np.ndarray:
        '''A static method that operates at the ChipSet-level and Chip-level to generate boolean
        masks where a value of 'True' indicates the presence of a NoData Value (ndv).

        TODO:
        1) Create unit tests to ensure this operation performs.
        '''
        try:
            chip_masks = np.all(arr == ndv, axis=(3))
            chipset_mask = np.all(arr == ndv, axis=(1, 2, 3))
            return chip_masks, chipset_mask
        except Exception as e:
            print(f"An unexpected error occured when attempting to generate NoData Masks: {e}.")
            raise

    def stack_chips(self, chip_hw: tuple[int, int] | int = 512, dtype: np.dtype = 'GeoChipSet') -> None:
        """ This is an ultra-fast, full-featured chipping function adapted from this
        article: https://archive.ph/eV7tq.

        This function "chips" a large image into many contiguous smaller chunks (which we
        call "chips") based on a user-specified chip_hw size. Chipping is an essential step
        for geospatial object detection since images of the earth tend to be quite large
        (~10,000 x 10,000 pixels or more), wheras most object detection models tend to
        prefer images that are much smaller in size (~1,000 x 1000 pixels or less).

        Since this function is primarily concerned with geospatial images, special
        consideration are made. First, the image is padded with user-specified no data
        values (NDVs) along the right and bottom edges. This padding ensures an even
        number of chips are produced, without the need for image resampling, which could
        result in the loss of small features or feature distortion.

        Once object detection has been performed on this function's chips, the ML results can be
        passed to the "unstack_chips" function to easily translate from chip image coordinates to
        source image coordinates.

        Inputs:
        - self.array: a numpy array of the image to be chipped (along with image
                            metadata such as shape, dtype, etc.).
        - chip_hw: a int OR tuple of two ints (height, width) that defines the desired chip
            size (in pixels). Example: (512, 512) produces chips of size 512 x 512 pixels.

        Returns:
        - self.chipset: a numpy array that contains each image chip along the first axis.
                        The shape is (num_chips, chip_height, chip_width, chip_bands).
        - self.chipset_tl_anchors: a numpy array that contains the source image coordinates for
                        top-left corner pixel of every chip image. Shape: (num_chips, row, col).
        - self.chipset_mask: a numpy array that contains a boolean mask for each chip that
                        denotes if the chip contains entirely NDV values. Shape: (num_chips,).

          TODO:
          - Add a "verbose" parameter to print out more information.
          - Add OVERLAP! This would be most desired (I think).
          - Masked arrays?
          - Make this work with a chw order instead of hwc.
        """
        if dtype == "GeoChipSet":
            dtype = self.GEOCHIPSET_DTYPE
        else:
            raise ValueError(f"Invalid ChipSet dtype selected: {dtype}.")

        img_h, img_w, img_bands = self.src_shape
        ndv = 0 if self.src_nodata is None else self.src_nodata  # Pad images w/ 0 if no NDV

        # Kernels of type int are re-formatted as a tuple of ints.
        if isinstance(chip_hw, int):
            chip_hw = (chip_hw, chip_hw)

        self.chip_hw = chip_hw
        chip_h, chip_w = self.chip_hw

        # Determine the number of chips needed to cover source image along each axis (height/width).
        self.num_chips_hw = (math.ceil(img_h / chip_h),
                             math.ceil(img_w / chip_w))
        num_chips_h, num_chips_w = self.num_chips_hw

        print(f"Kernel size of {chip_hw} with image of size ({img_h}, {img_w}) will result in"
              f" {num_chips_h * num_chips_w} output image chips... ")

        # Determine how much pixel padding needed along each axis of source image to fill evey chip.
        pad_h = (chip_h * num_chips_h) - img_h
        pad_w = (chip_w * num_chips_w) - img_w

        # If needed, pad the source image along the bottom and right sides with pixels of values set
        # to the input image's NDV. If no NDV is specified, the default value of "0" will be used.
        if pad_h == 0 and pad_w == 0:
            print("No padding neccecary...")
            padded_array_img = self.array_img
        else:
            print(f"Padding image by ({pad_h}, {pad_w})...")
            padded_array_img = np.pad(self.array_img,  # NOTE: this method requires PIL order: (h, w, c)
                                      [(0, pad_h), (0, pad_w), (0, 0)],
                                      mode="constant",
                                      constant_values=ndv)
        self.padded_src_shape = padded_array_img.shape

        print(f"Source Image Shape {self.array_img.shape}"
              f" | Padded Image Shape: {padded_array_img.shape}.")

        # This is where the "real" magic happens.
        chipped_array = padded_array_img.reshape(num_chips_h, chip_h,
                                                 num_chips_w, chip_w, img_bands)
        chipped_array = chipped_array.swapaxes(1, 2)

        # Reshape the chip array so all image chips are stacked along a single axis. This will reshape
        # to our final shape of: (num_chips, chip_hw height, chip_hw width, image img_bands).
        shaped_array = chipped_array.reshape(-1, *(chip_h, chip_w, img_bands))

        # Create an array of the top-left corner pixel coordinates in the same order as shaped_array.
        tops = np.array([y * chip_h for y in range(0, num_chips_h)]).astype('uint16')
        lefts = np.array([x * chip_w for x in range(0, num_chips_w)]).astype('uint16')
        tl_pairs = list(itertools.product(tops, lefts))
        tl_array = np.asarray(tl_pairs).astype('uint16')

        # Create an array of shape (num_chips,) that holds boolean values telling whether a chip
        # is comprised completely of NDVs. This can happen on especially large or irregularly
        # shaped aerial images.
        chip_masks, chipset_mask = self._mask_ndvs(shaped_array, ndv)

        self.num_chips = shaped_array.shape[0]

        # Set final attributes, done!
        self.chipset = np.empty(self.num_chips, dtype=dtype)
        for i in range(self.num_chips):
            self.chipset[i]['chip'] = shaped_array[i]
            self.chipset[i]['chip_mask'] = chip_masks[i]
            self.chipset[i]['topleft_anchor'] = tl_array[i]
            self.chipset[i]['is_invalid'] = chipset_mask[i]

        print("Geospatial chipping operation complete."
              f" Final chipped array shape: {shaped_array.shape}.")


""" NOTE: THE METHODS BELOW ARE RELATED TO REVERSING THE STACKING (MAPPING LABELS TO IRL COORDS.)
          CURRENTLY THESE METHODS ARE UNTESTED IN THIS SCRIPT (BUT KNOWN TO BE GENERALLY WORKING).
          THESE WILL BE TESTED AND INCORPORATED SOON...


    @staticmethod
    def _denormalize_coordinates(b_box: list[float, float, float, float],
                                 chip_hw: tuple[int, int]) -> list[int, int, int, int]:
        '''TODO: Check documentation, unit test.

        Converts normalized coordinates (0.0 - 1.0) to image pixel coordinates.

        Input bounding box coordinates should be in Tensorflow's preferred coordinate
        order of (ymin, xmin, ymax, xmax). The return coordinates are in the same order.

        Args:
            bbox (list[floats]): The normalized bounding box coordinates in the order
                (ymin, xmin, ymax, xmax).
            chip_hw (tuple[ints]): The height and width of the chip image.

        Returns:
            list[ints]: The image pixel coordinates of the bounding box in the order
                (ymin, xmin, ymax, xmax).
        '''

        # this is set to Tensorflow Object Detection ordering (ymin, xmin, ymax, xmax)
        ymin, xmin, ymax, xmax = b_box
        chip_h, chip_w = chip_hw

        top = int(ymin * chip_h)
        left = int(xmin * chip_w)
        bottom = int(ymax * chip_h)
        right = int(xmax * chip_w)

        return [top, left, bottom, right]

    def unstack_chips(self, inference_results_dict):
        '''TODO: Re-write code to not be *so* TF specific, unit test, write docs. This code
        is completely provisional.'''

        merged_results_dict = {}
        new_bboxes = []
        new_scores = []
        new_classes = []

        for i, i_results in inference_results_dict.items():
            y_offset = self.chipping_topleft_array[i][0]
            x_offset = self.chipping_topleft_array[i][1]

            for normalized_bbox in i_results["detection_boxes"]:
                bbox = self._denormalize_coordinates(normalized_bbox)
                new_bbox = (
                    int(y_offset + bbox[0]),
                    int(x_offset + bbox[1]),
                    int(y_offset + bbox[2]),
                    int(x_offset + bbox[3]),
                )
                new_bboxes.append(new_bbox)

            for score in i_results["detection_scores"]:
                new_scores.append(score)

            for classes in i_results["detection_classes"]:
                new_classes.append(classes)

        merged_results_dict[self.file_name] = {
            "bboxes": new_bboxes,
            "scores": new_scores,
            "classes": new_classes,
        }

        return merged_results_dict
"""
