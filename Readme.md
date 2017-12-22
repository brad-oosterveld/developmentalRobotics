Comp 150- Developmental Robotics

Extending Instruction-Based One-Shot Learning in a CRA by generalizing from a few examples

Bradley Oosterveld & Tyler Frasca

1. Installation

* Inception-v3:
    * Follow instructions found here to install Tensorflow and the Inception-v3 model: https://www.tensorflow.org/tutorials/image_recognition

* VFH:
    * Follow http://wiki.ros.org/kinetic/Installation/Ubuntu to install ROS
    * ```$ sudo apt-get install ros-VERSION-pcl-ros ros-VERSION-pcl-msgs```
         *Change VERSION to version of ROS
         
2. Sub-sample Dataset:
    * Download dataset: http://rgbd-dataset.cs.washington.edu/dataset/
    * update paths in extract images.py
    * ```$ python extract images.py```
    * ```$ python compareFiles.py```
    * remove entries that only appear in one dataset
    
3. Extract Features:
* 2D
    * ```$ cp get_inception_features.py to $TF_INSTALL_DIR/models/tutorials/image/imagenet```
    * ```$ cd $TF_INSTALL_DIR/models/tutorials/image/imagenet```
    * ```$ python get_inception_features.py $PATH_TO_DATA```
* 3D
    * ```$ cd catkin_ws```
    * ```$ catkin_make```
    * ```$ source devel/setup.bash```
    * ```$ roscore```
    * ```$ rosrun object_classification object_classification_node``` 

4. Run Evaluations:
    * ```$ python eval.py```

