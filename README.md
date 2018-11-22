# SRCNN
This project has implemented a SRCNN model using Pytorch for improve the resolution of images.

## Requirements
 - torch 0.4.1
 - scikit-learn 0.19.1
 - yagmail 0.11.214
 - plotly 2.0.11
 - Python 3.6 and above

## Dataset description
 - train data: 64x64 image and 128x128 image pairs
 - test data: 64x64 image
 - __Note: this script only works for image size of 64x64 and 128x182 now. Waiting to be updated later__

## Parameter explaination
 - Run `python main.py -h` and will get options on parameters you can set for this script. 
 
## Algorithm explanation
 - Use [bicubic](http://www.paulinternet.nl/?page=bicubic) interpolation to interpolating image size of 64x64 to size of 128x128
 - Use SRCNN to improve the resolution of image size of 128x128


# reference
[1] Dong, Chao, et al. "Image super-resolution using deep convolutional networks." IEEE transactions on pattern analysis and machine intelligence 38.2 (2016): 295-307. [PDF](https://arxiv.org/pdf/1501.00092.pdf)
