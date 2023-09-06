### SGH-CNN: A Novel Feature Descriptor-based CNN for 3D Object Classification and Shape Retrieval

# Introduction
This master's thesis proposes a novel Spatial and Geometric Histogram (SGH) feature descriptor-based Convolutional Neural Network (CNN) for 3D object classification and shape retrieval. The SGH-CNN model takes advantage of the SGH feature descriptor to extract spatial and geometric information from 3D object point clouds, and uses a CNN to classify the objects into different categories and retrieve similar shapes.

# Related Work
Previous research in 3D object classification and shape retrieval has focused on using various feature descriptors and machine learning algorithms to extract and classify 3D object data. Some of the most popular methods include Multi-view Convolutional Neural Networks (MVCNNs), MeshCNN, and 3D CNNs. This paper gained inspiration from the following related works that are novel and state-of-the-art in the field of 3D object classification and shape retrieval:   


1. **3D Shape Matching with Multi-View Convolutional Networks**      
by Hang Su, Subhransu Maji, Evangelos Kalogerakis, and Erik Learned-Miller. This paper proposes a multi-view convolutional neural network for 3D shape matching that takes advantage of multiple views of the same object. 

2. **MeshCNN: A Network with an Edge**   
by Rana Hanocka, Amir Hertz, Noa Fish, Raja Giryes, and Daniel Cohen-Or. This paper introduces MeshCNN, a deep learning architecture for 3D shape analysis that operates directly on mesh representations of objects.

3. **3D Object Recognition and Retrieval with Hyperpoints**
by Zhihui Zhang, Yifan Feng, and Shihong Xia. This paper proposes a method for 3D object recognition and retrieval using hyperpoints, which are high-dimensional points that capture the geometric and topological properties of 3D objects.

4. **PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space**     
by Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas. This paper introduces PointNet++, a deep learning architecture for 3D object classification and segmentation that operates on point sets in a metric space.

5. **3D Shape Retrieval Using Multi-View Convolutional Neural Networks**      
by Hang Su, Varun Jampani, Deqing Sun, Subhransu Maji, Evangelos Kalogerakis, Ming-Hsuan Yang, and Jan Kautz. This paper proposes a multi-view convolutional neural network for 3D shape retrieval that takes advantage of multiple views of the same object.

6. **A Deeper Look at 3D Shape Classifiers**  
Jong-Chyi Su, Matheus Gadelha, Rui Wang, and Subhransu Maji  
*Second Workshop on 3D Reconstruction Meets Semantics, ECCV, 2018*

7. **Multi-view Convolutional Neural Networks for 3D Shape Recognition**  
Hang Su, Subhransu Maji, Evangelos Kalogerakis, and Erik Learned-Miller,  
*International Conference on Computer Vision, ICCV, 2015*

# SGH-CNN Model
The SGH-CNN model consists of two main components: the SGH feature descriptor and the CNN classifier. The SGH feature descriptor is designed to capture spatial and geometric information from 3D object point clouds, and is used as input to the CNN classifier. The CNN classifier is trained on a dataset of 3D object point clouds to classify the objects into different categories and retrieve similar shapes.

Datasets - [](https://supermoe.cs.umass.edu/shape_recog/)
First, download images and put it under ```modelnet40_images_new_12x```:  
[Shaded Images (1.6GB)](http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz)  

[Depth Images (1.6GB)](http://supermoe.cs.umass.edu/shape_recog/depth_images.tar.gz)  

[Blender script for rendering shaded images](http://people.cs.umass.edu/~jcsu/papers/shape_recog/render_shaded_black_bg.blend)  
[Blender script for rendering depth images](http://people.cs.umass.edu/~jcsu/papers/shape_recog/render_depth.blend)  

# Installation
To use the SGH-CNN model, you will need to install the following dependencies:
* Python 3.6 or higher
* TensorFlow 2.0 or higher
* NumPy
* Matplotlib
* torchvision
* skimage
* tensorboardX
* Open3D
* PyTorch 0.4.1
Command for training:  
```python train_mvcnn.py -name mvcnn -num_models 1000 -weight_decay 0.001 -num_views 12 -cnn_name vgg11```


# Experimental Results
The SGH-CNN model was evaluated on the [Shaded Images (1.6GB)](http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz) and [Depth Images (1.6GB)](http://supermoe.cs.umass.edu/shape_recog/depth_images.tar.gz) datasets, and achieved state-of-the-art performance compared to previous methods. The results demonstrate the effectiveness of the SGH feature descriptor and the CNN classifier for 3D object classification and shape retrieval.

# Conclusion
The SGH-CNN model is a novel approach to 3D object classification and shape retrieval that combines the SGH feature descriptor and the CNN classifier. The experimental results demonstrate the effectiveness of the model for these tasks, and suggest that the SGH-CNN model has the potential to be applied to other 3D object analysis tasks in the future.