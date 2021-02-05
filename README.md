# Feature based image registrator

The image registrator uses `FAST` feature finder and `DAISY` feature descriptor for registration. 
It can align images of different size by padding them with 0 values. 
The image registrator can work with multichannel grayscale TIFFs and TIFFs with multiple z-planes. 
Images **MUST** have OME-TIFF XML in their description.
The script does tile based registration of images to avoid big memory consumption. 

## Command line arguments

**`-i`**    paths to images you want to register separated by space.

**`-r`**    reference image id, e.g. if `-i 1.tif 2.tif 3.tif`, and you ref image is `1.tif`, then `-r 0` (starting from 0)

**`-c`**    reference channel name, e.g. DAPI. Enclose in double quotes if name consist of several words e.g. "Atto 490LS".

**`-o`**    directory to output registered image.

**`-n`**    multiprocessing: number of processes, default 1

**`--stack`**  add this flag if input is image stack instead of image list

**`--estimate_only`**   add this flag if you want to get only registration parameters and do not want to process images.

**`--load_param`**  specify path to csv file with registration parameters


## Example usage

`python reg.py -i "/path/to/image1/img1.tif" "/path/to/image2/img2.tif" -o  "/path/to/output/directory" -r 0 -c "Atto 490LS" -n 3`

## Dependencies

`numpy pandas tifffile opencv-contrib-python scikit-image`

`scikit-image` is necessary for affine transformation of big images that has more than 32000 pixels in one or two dimensions. 
The affine registration process in `scikit-image` requires usage of `float64` data, so you need amount of RAM at least 3 times the size of the picture (channel, z-plane).