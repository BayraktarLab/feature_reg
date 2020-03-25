# Feature based image registrator

The programs uses `FAST` feature finder and `DAISY` feature descriptor for registration. It can align images of different size by padding them with 0 values. The image registrator can work with multichannel grayscale TIFFs and TIFFs with multiple z-planes. Images must have OME-TIFF XML in their description.

## Command line arguments

**`-i`**    paths to images you want to register separated by space.

**`-r`**    reference image id, e.g. if `-i 1.tif 2.tif 3.tif`, and you ref image is `1.tif`, then `-r 0` (starting from 0)

**`-c`**    reference channel name, e.g. DAPI. Enclose in double quotes if name consist of several words e.g. "Atto 490LS".

**`-o`**    directory to output registered image.')

**`-s`**    scale of the images during registration in fractions of 1. 1 - full scale, 0.5 - half scale. Default value is 0.5.

**`--estimate_only`**   add this flag if you want to get only registration parameters and do not want to process images.

**`--load_param`**  specify path to csv file with registration parameters


## Example usage

`python reg.py -i "/path/to/image1/img1.tif" "/path/to/image2/img2.tif" -o  "/path/to/output/directory" -r 0 -c "Atto 490LS" -s 0.5`

## Dependencies

`numpy pandas tifffile opencv-contrib-python`
