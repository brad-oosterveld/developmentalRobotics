Comp 150- Developmental Robotics

Extending Instruction-Based One-Shot Learning in a CRA by generalizing from a few examples

Bradley Oosterveld & Tyler Frasca

1. Installation

* Inception-v3:
    * Follow instructions found here to install Tensorflow and the Inception-v3 model: https://www.tensorflow.org/tutorials/image_recognition

* VFH:

2. Sub-sample Dataset:

3. Extract Features:
* 2D
    * ```$ cp get_inception_features.py to $TF_INSTALL_DIR/models/tutorials/image/imagenet```
    * ```$ cd $TF_INSTALL_DIR/models/tutorials/image/imagenet```
    * ```$ python get_inception_features.py $PATH_TO_DATA```

4. Run Evaluations:

