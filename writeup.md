# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images_output/bar_per_Category.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/image1.png "Traffic Sign 1"
[image5]: ./test_images/image2.png "Traffic Sign 2"
[image6]: ./test_images/image3.png "Traffic Sign 3"
[image7]: ./test_images/image4.png "Traffic Sign 4"
[image8]: ./test_images/image5.png "Traffic Sign 5"
[image9]: ./images_output/lenet.png "network architecture"
[image10]: ./test_images/image6.png "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43** 

#### 2. Include an exploratory visualization of the dataset.

Here's an exploratory visualization of the dataset. A bar graph where the y-axis is located vertically on the left side has 43 signs and traffic classes. On the other hand, we have the x-axis horizontal that describes each class's number of traffic signs (0 - 2000).

The bar graph shows the distribution of the classification of traffic signs. The color blue long bars indicate the classes that stand out the most from those between 1750 and 2000. For example, let's look at the axis, and the class of "Speed limit ( 80km / h) "and End of no pass" are the classes with the highest number of signals, very close to 2000 according to the graph.

Now let's see the classification with the color red shortest bars and that they are around the value of 250. For example, we have the class "Speed limit (20km / h)" and "Speed limit (120km / h). They are above that number.

The graph shows us a visual way to identify how our dataset is distributed quickly.


![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

It is necessary to perform some transformations on each image to feed it to the neural network.

The pre-processing was for each of the 3 color channels (RGB); for image data, (pixel - 128)/ 128 is used a quick way to approximately normalize the data and can be used for this project


Normalize the RGB image. It is done to make each image "look similar" to each other so that the input is consistent.
Convert RGB image to grayscale. The neural network is believed to perform slightly better on grayscale images. However, they can also be incorrect observations.
An attempt was also made to use adaptive histogram equalization to improve local contrast and improve edge definitions in each image region. Still, it decreased network performance, So it is the only normalization and grayscale conversion used in the final implementation.


Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]


The training set expanded with more data. The intention was to equalize the sample count in each category, the categories containing a smaller number of samples expanded with more duplicate images. The odds of getting images for each category during training became equal. Dramatically improved neural network performance. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is based off LeNet consisted of the following layers:

##### Architecture

| Layer           |     Description 	                              | 
|:----------------|:--------------------------------------------------| 
| Input           | 32x32x3 RGB image                                 |
| Convolution 5x5 | 1x1 stride, `VALID` padding, outputs 28x28x6      |
| ReLU            |                                                   |
| Max pooling     | k=2, 2x2 stride, `SAME` padding, outputs 14x14x16 |
| ReLU            |                                                   |
| Convolution 5x5 | 1x1 stride, `VALID` padding, outputs 10x10x16     |
| ReLU            |                                                   |
| Max pooling     | k=2, 2x2 stride, `SAME` padding, outputs 5x5x16   |
| Flatten   	  | outputs 400                                       |
| Fully connected | outputs 120                                       |
| ReLU            |                                                   |
| Fully connected | outputs 84                                        |
| ReLU            |                                                   |
| Fully connected | outputs 43                                        |
 




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

50 epochs were used to train the model. Initially, 10 was used; however, discovered that more epochs were needed to achieve greater precision. Also tested it 100 epochs, deduced that it did not improve over 50 epochs.

The same optimizer, tf.train.AdamOptimizer was used as in the LeNet lab.

I used rate = 0.001 for the learning rate; I tried using rate = 0.0001; however, the results weren't that good.

For attrition, keep_prob = 0.9 was used when training, found to offer better results than keep_prob = 0.5



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

A LeNet model was used where it was modified to accept inputs with a depth of 3 (32x32x3) instead of 1 (32x32x1), as we are working with RGB images and not grayscale images.
Changed the length of the output to 43 (the number of classifications), not 10 (the number of digits). Also introduced dropout into the model to achieve greater precision. 

LeNet model presented in his 1998 research paper using TensorFlow. The algorithm architecture consists of 2 convolution layers followed by three fully connected layers. Convolution layers handle the extraction of features from traffic sign images, which help learn to recognize them. Each convolution layer has followed by relu activation and maximum grouping to filter images down to the pixels that matter. After the last convolution layer, the flatten operation has applied to transform the output shape to a vector. The first two fully connected layers had followed by reluctance and dropout to prevent the model from overfitting the data. Finally, the fully connected layers reduce the output vector to 43 different classes so that the model can predict the type of road sign in the image. Can find the code implementation for the modified LeNet architecture function in the Jupyter notebook. 

The final model results were:

* training set accuracy of 100.0%
* validation set accuracy of 95.6%
* test set accuracy of 94.8%

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The algorithm design aims to improve the identification of traffic signs found on the streets and highways through the rapid acquisition and interpretation of the images. However, several non-technical external challenges that this system can face in the real world, degrading its performance significantly. Among these challenges are; Variable lighting conditions and fading and blurring effects caused by lighting through rain or snow.

Here are five German traffic signs that I found on the web:

<img src="./test_images/9_no_passing.jpg " alt="data_augumentation" width="200"/>

<img src="./test_images/11_right-of-way.jpg " alt="data_augumentation" width="200"/>

<img src="./test_images/12_priority_road.jpg " alt="data_augumentation" width="200"/>

<img src="./test_images/15_no_vehicles.jpg " alt="data_augumentation" width="200"/>

<img src="./test_images/18_general_caution.jpg " alt="data_augumentation" width="200"/>



The second and third images could represent difficulties to classify due to low light and show additional features of shapes in the background that the algorithm misinterpreted.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                                      | Prediction                                 |
|:-------------------------------------------|:-------------------------------------------| 
| (9) No passing                             | (9) No passing                             | 
| (11) Right-of-way at the next intersection | (11) Right-of-way at the next intersection | 
| (12) Priority road                         | (12) Priority road                         |
| (15) No vehicles                           | (15) No vehicles                           |
| (18) General caution                       | (18) General caution                       |



The model was able to correctly predicted 5 of the 6 road signs, giving an accuracy of 94.8%. However, on occasions, it does not show the same precision. It is to say that when the algorithm is run, there are times that it only asserts 4 out of 5.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99678e-01         			| No passing   									| 
| 1.00000e+00     				| Right-of-way at the next 										|
| 
1.00000e+00					| Priority road											|
| 1.00000e+00	      			| No vehicles					 				|
| 1.00000e+00				    | General caution      							|


![test_set_distribution.png](images_output/five_softmax_probabilities.png)

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


