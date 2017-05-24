#**Traffic Sign Recognition** 

##Writeup by Ernesto Cañibano

###This is the writeup associated with the second project of the Term 1, of the Self-driving Car Nanodegree

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

[image1]: ./writeup-images/train_graphic_bar.png "Train dataset graphic bar"
[image2]: ./writeup-images/dataset-examples.png "Train dataset examples"
[image3]: ./writeup-images/normalization-example.png "Normalization example"
[image4]: ./extra-signs/1.jpg "Sign 1"
[image5]: ./extra-signs/14.jpg "Sign 14"
[image6]: ./extra-signs/18.jpg "Sign 18"
[image7]: ./extra-signs/22.jpg "Sign 22"
[image8]: ./extra-signs/25.jpg "Sign 25"
[image9]: ./extra-signs/27.jpg "Sign 27"
[image10]: ./extra-signs/28.jpg "Sign 28"
[image11]: ./extra-signs/32.jpg "Sign 32"
[image12]: ./extra-signs/38.jpg "Sign 38"
[image13]: ./extra-signs/40.jpg "Sign 40"
[image14]: ./writeup-images/softmax.png "Softmax probabilities"
[image15]: ./writeup-images/activations.png "Activations"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ernestocanibano/CarND-Traffic-Sign-Classifier-Project-P2/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. For it I did a bar char showing the number of signals of each class. I also show five signals of each class as an example. The five images are selected ramdonly.

I processed the dataset independently, repeating the process for the train dataset, the validation datasest and the test dataset. In the following images it is possible to see the bar chart for train dataset and some of the signals.

![alt text][image1]

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided not to convert the images to grayscale because the results with my model were similar working in color and grayscale.

I normalized the image data because the images have values between 0 and 255. To have the inputs well contiditioned I normalized these values to achieve a range between -1 and 1 wich have mean 0.

Here is an example of a traffic sign image before and after normalization.

![alt text][image3]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is based in LeNet-5 modified, it consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32   |
| RELU 					| 2x2 stride,  outputs 5x5x32  					|
| Fully connected		| outputs 1x256        							|
| Dropout       		| 0.5        									|
| Fully connected		| outputs 1x128        							|
| Dropout				| 0.5											|
| Fully connected		| outputs 1x43									|
| Softmax				| 			  	       							|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following:

* Batch size = 128. I tried several values and 128 worked properly.
* Epochs = 10. I tried with 10 and achieve a 96% of accuracy. I tested with bigger values but accuracy began to decrease (overfitting).
* Learning rate = 0.001. With this value I achieved a good accuracy and an acceptable trainin time.
* Optimizer = Adam Optimizier
* Dropout = 0.5. I tried several values with the following results: 1.0 = 91%, 0.75 = 95%, 0.5 = 96%, 0.25 = 90%. At the beginning I didn't include dropout in the model, when I include the dropout I achieved to improve the accuracy.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99%.
* validation set accuracy of 96%. 
* test set accuracy of 95%.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11] ![alt text][image12]
![alt text][image13]

The images 1, 3, 4 might be difficult to classify because they have additional signs near the main signal.

The images 6, 7, 9 might be problems with the classification because they have a watermark.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 30km/h	| Right-of-way at the next intersection			| 
| Stop	     			| Stop 											|
| General caution		| Slippery road									|
| Bumpy road	      	| Road work					 					|
| Road work				| Road work      								|
| Pedestrians			| General caution      							|
| Children crossing		| Right-of-way at the next intersection      	|
| End of all speed and passing limits	| End of all speed and passing limits |
| Keep right			| Keep right      								|
| Roundabout mandatory	| Roundabout mandatory 							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 50%. 

As I had anticipated there is a problem with the images with watermarks. Maybe the accuracy with this type of signals could increase training the model with signal mixed with noisy.

To improve the accuracy with signals which have a signboard below them It would be neccessary train the model with more signals of this type.


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 29th cell of the Ipython notebook.

![alt text][image14]

For the images 2, 5, 9 y 10 the model is sure (probability of 0.99), and the prediction match the sign right. 

For the image 8 (End of all speed and passing limits) the model is not much sure (probability of 0.28) but the prediction match the right sign.

For the image 5 (Pedrestians) the model fails and detect another sign (General caution) with a very high probability (0.99). It could be because the model was trained with high number of "General caution" signals and a low number of "Pedrestians" signals.

 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image15]
