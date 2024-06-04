"""Unit tests for ChipSet.py."""
import tempfile
import unittest
import warnings
from itertools import islice

import numpy as np
import rasterio as rio
from rasterio.transform import from_origin
from rasterio.errors import NotGeoreferencedWarning

from ChipSet import GeoChipSet


class TestGeoChipSet(unittest.TestCase):
    def setUp(self):
        '''Initialize a checkerboard array of shape (9,9,3). When chipped using a (3x3) kernel the
        the resulting chips should be "All Ones", "All Twos", etc. The one difference is the
        "Nines" chip contains a single 0 value in the bottom right corner. This is to simulate a no
        data value, and ensure those are handled appropriately.
        '''
        print("\n**** Beginning test of class GeoChipSet() **************************************")
        self.checkerboard = np.array([
            [[1, 1, 1, 2, 2, 2, 3, 3, 3],
             [1, 1, 1, 2, 2, 2, 3, 3, 3],
             [1, 1, 1, 2, 2, 2, 3, 3, 3],
             [4, 4, 4, 5, 5, 5, 6, 6, 6],
             [4, 4, 4, 5, 5, 5, 6, 6, 6],
             [4, 4, 4, 5, 5, 5, 6, 6, 6],
             [7, 7, 7, 8, 8, 8, 9, 9, 9],
             [7, 7, 7, 8, 8, 8, 9, 9, 9],
             [7, 7, 7, 8, 8, 8, 9, 9, 0]],
            [[1, 1, 1, 2, 2, 2, 3, 3, 3],
             [1, 1, 1, 2, 2, 2, 3, 3, 3],
             [1, 1, 1, 2, 2, 2, 3, 3, 3],
             [4, 4, 4, 5, 5, 5, 6, 6, 6],
             [4, 4, 4, 5, 5, 5, 6, 6, 6],
             [4, 4, 4, 5, 5, 5, 6, 6, 6],
             [7, 7, 7, 8, 8, 8, 9, 9, 9],
             [7, 7, 7, 8, 8, 8, 9, 9, 9],
             [7, 7, 7, 8, 8, 8, 9, 9, 0]],
            [[1, 1, 1, 2, 2, 2, 3, 3, 3],
             [1, 1, 1, 2, 2, 2, 3, 3, 3],
             [1, 1, 1, 2, 2, 2, 3, 3, 3],
             [4, 4, 4, 5, 5, 5, 6, 6, 6],
             [4, 4, 4, 5, 5, 5, 6, 6, 6],
             [4, 4, 4, 5, 5, 5, 6, 6, 6],
             [7, 7, 7, 8, 8, 8, 9, 9, 9],
             [7, 7, 7, 8, 8, 8, 9, 9, 9],
             [7, 7, 7, 8, 8, 8, 9, 9, 0]]]).astype(float)
        self.checkboard_shape = self.checkerboard.shape
        self.checkerboard_nodata = 0

        try:
            self.rio_checker = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            transform = from_origin(1, 1, 1, 1)  # Define a simple transform
            with rio.open(self.rio_checker.name, 'w', driver='GTiff', height=self.checkboard_shape[1],
                          width=self.checkboard_shape[2], count=3, dtype=self.checkerboard.dtype,
                          nodata=self.checkerboard_nodata, transform=transform) as dst:
                 dst.write(self.checkerboard)

            self.geochipset = GeoChipSet(self.rio_checker.name)
        except Exception as e:
            print(f"Unexpected error while creating test dataset: {e}.")

    def test_stack_chips_3x3(self):
        '''Test the stack_chips function to ensure the checkerboard array is chipped correctly.'''
        print("\n**** GeoChipSet Test 1: Basic Chip Stacking (3x3) ******************************")
        self.geochipset.stack_chips((3, 3))

        expected_chips = [np.ones((3, 3, 3)), np.ones((3, 3, 3)) * 2, np.ones((3, 3, 3)) * 3,
                          np.ones((3, 3, 3)) * 4, np.ones((3, 3, 3)) * 5, np.ones((3, 3, 3)) * 6,
                          np.ones((3, 3, 3)) * 7, np.ones((3, 3, 3)) * 8, np.ones((3, 3, 3)) * 9]
        expected_chips[-1][:, -1, -1] = 0  # The last chip contains a single 0 value in the bottom right corner

        for i, chip in enumerate(islice(self.geochipset.yield_chips(), 0, 7)):
            with self.subTest(i=i):
                np.testing.assert_array_equal(chip['chip'], expected_chips[i], err_msg=f"Chip {i} is not as expected.")


if __name__ == '__main__':
    unittest.main()
