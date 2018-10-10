# Python2 Mask R-CNN for Object Detection and Segmentation

This is a fork of [Matterport's Mask R-CNN implementation](https://github.com/matterport/Mask_RCNN) modified to function with python2. It has not yet been tested to see if the model still trains or not, and I've definitely broken some functionality (downloading pretrained models, etc.), but if you just need image segmentation with a coco-pretrained model in python2, you're in luck.

## Installation
1. Clone the repository.
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Optional: Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ```
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases), place it in the root directory of this package.

4. Execute ```python pretrained_test.py``` with a python2 interpreter! This program uses PretrainedLoader to construct and load a model, then evaluates it on the images provided with the package.

To include this in your project, ```import PretrainedLoader``` as in pretrained_test.py.

Happy segmenting!
