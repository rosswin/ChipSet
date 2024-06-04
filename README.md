# ChipSet
Build better computer vision pipelines for geospatial data by splitting large images into sets of smaller images (or "chips") using ChipSet!

## Description
ChipSet includes tools for "chipping" large geospatial images into small, contiguous chunks so that computer vision algorithms can focus on 
one small patch. Efficient chipping operations are critical when working with large batches of 10,000 x 10,000 pixel satellite images (or larger!) and the YOLO classifier works on images of size 640 x 640 pixels.

In addition to seamlessly and rapidly chipping images in a memory-efficient manner, ChipSet aspires to support the following features:
- The ability to reverse the chipping operation -- seamlessly convert computer vision results performed on chips back the original image.
- The ability to view intermediate chipping steps and the impact of different chipping strategies (overlap, no overlap, kernel size, etc.)
- Support for basic RGB images, in addition to multi-band satellite images.
- Support for multiple geospatial metadata formats: orthorectified, georeferenced, EXIF GPS information, and/or no geospatial meadata.
- Support for basic geospatial software components: __GeoInterface__, writing to GIS vectors, etc.
- SuperChipSets: the ability to read folders of diverse image types into a single "container" for ChipSets that allows batch processing
    the imagery and analyzing the computer vision results in a seamless workflow.
- Support for high-efficiency file formats such as HDF, Zarr (or GeoZarr). This would allow SuperChipSets to keep their intermediate
    data on disk, and read it only what is needed when needed.
- Abiilty to persist SuperChipSets/ChipSets on-disk and re-initialize in future workflows. Chip once, persist data, re-use for future analyses.
- and more...

## Installation
Right now `git clone` and creating an environment from `requirements.txt` is the only option.

## Usage
**COMING SOON**
Full usage instruction coming soon, but in the meantime the `Example_GeoChipSet.ipynb` Notebbok is included as an example of the high-level workflow.


## Testing
Basic unit testing is included for the core operations. More testing to come in the future. The following command will run the full test suite:

```python
python -m unittest discover -s tests
```
