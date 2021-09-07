{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data Set Summary & Exploration\n",
    "\n",
    "#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.\n",
    "\n",
    "I used the len() method to calculate summary statistics of the traffic signs data. To calculate number of classes, I used len(set()) to get number of classes in dataset\n",
    "\n",
    "* Number of training examples = 34799\n",
    "* Number of training examples = 4410\n",
    "* Number of testing examples = 12630\n",
    "* Image data shape = (32, 32, 3)\n",
    "* Number of classes = 43\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2. Include an exploratory visualization of the dataset.\n",
    "\n",
    "I chose 15 random images from image dataset. Subset of training dataset with labels can be shown below:\n",
    "\n",
    "![alttext](output_images/subset.jpg)\n",
    "\n",
    "Here is an exploratory visualization of the data set. It is a bar chart showing how the frequency of an image in dataset on y axis.\n",
    "\n",
    "![alttext](output_images/dataExploration.jpg)\n",
    "\n",
    "#### 2. Preprocessing of Image Data\n",
    "Each image in dataset (train, test and validation) has been converted into greyscale by average the R,G,B pixels and then normalizing the entire data set by subtracting pixel value by 128 and then dividing by 128. I have not augmenting the data, since the data provided was enough to prevent overfitting and achieve validation accuracy of 96%. Output after converting to grayscale and normalization can be shown below:\n",
    "\n",
    "![alttext](output_images/processed.jpg)\n",
    "\n",
    "### Design and Test a Model Architecture\n",
    "\n",
    "#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.\n",
    "\n",
    "I used LeNet with dropout. My final model consisted of the following layers:\n",
    "\n",
    "| Layer         \t\t|     Description\t        \t\t\t\t\t| \n",
    "|:---------------------:|:---------------------------------------------:| \n",
    "| Input         \t\t| 32x32x3 RGB image   \t\t\t\t\t\t\t| \n",
    "| Convolution 5x5     \t| 1x1 stride, valid padding \t                |\n",
    "| RELU\t\t\t\t\t|\t\t\t\t\t\t\t\t\t\t\t\t|\n",
    "| Max pooling\t      \t| 2x2 stride, valid padding\t\t\t\t        |\n",
    "| Convolution 5x5\t    | 1x1 stride, valid padding      \t\t\t\t|\n",
    "| RELU           \t\t|            \t\t\t\t\t\t\t\t\t|\n",
    "| Max pooling    \t\t| 2x2 stride, valid padding        \t\t\t    |\n",
    "| Flatten\t\t        |        \t\t\t\t\t\t\t\t\t    |\n",
    "| Dropout        \t\t| keep_prob=0.7        \t\t\t\t\t\t\t|\n",
    "| Fully connected\t\t|         \t\t\t\t\t\t\t\t\t    |\n",
    "| Fully connected\t\t|                \t\t\t\t\t\t\t    |\n",
    "| Fully connected\t    |\t\t\t\t\t\t\t\t\t\t\t\t|\n",
    "| Softmax\t\t\t\t|\t\t\t\t\t\t\t\t\t\t\t\t|\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.\n",
    "\n",
    "To train the model, I used following parameters:\n",
    "    * learning rate=0.001\n",
    "    * Probability of keeping a node while using droput = 0.7\n",
    "    * Batch size = 128\n",
    "    * Number of Epochs = 50\n",
    "    * Optimizer = Adam\n",
    "\n",
    "#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.\n",
    "\n",
    "My final model results were:\n",
    "* validation set accuracy of 95.8% \n",
    "* test set accuracy of 93.8%\n",
    "\n",
    "Details of architecure:\n",
    "* I wanted to start with a known architecture to decrease number of iterations to be done. So, I started with basic LeNet architecture. I was able to get accuracy of 89%\n",
    "* Then, I decreased learning rate to 0.001 and increased number of epochs to 50. I was able to get validation accuracy of 93%\n",
    "* After that, I added dropout with probability of 0.7. Doing this, I was able to achieve validation accuracy of 96%"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Test a Model on New Images\n",
    "\n",
    "#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.\n",
    "\n",
    "Here are five German traffic signs that I found on the web:\n",
    "\n",
    "![alt text](output_images/germanSigns.jpg)\n",
    "\n",
    "The first image might be difficult to classify because ...\n",
    "\n",
    "#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the \"Stand Out Suggestions\" part of the rubric).\n",
    "\n",
    "Here are the results of the prediction:\n",
    "\n",
    "| Image\t\t\t                                |     Prediction\t        \t\t\t\t\t| \n",
    "|:---------------------------------------------:|:---------------------------------------------:| \n",
    "| 1            Speed limit (30km/h)      \t\t| 1            Speed limit (30km/h)   \t\t\t| \n",
    "| 11  Right-of-way at the next intersection     | 11  Right-of-way at the next intersection\t\t|\n",
    "| 15               No vehicles\t\t\t\t\t| 15               No vehicles\t\t\t     \t|\n",
    "| 12              Priority road\t      \t\t    | 12              Priority road   \t\t\t\t|\n",
    "| 14                  Stop\t\t\t            | 14                   Stop      \t\t\t\t|\n",
    "\n",
    "\n",
    "The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93%\n",
    "\n",
    "#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the \"Stand Out Suggestions\" part of the rubric, visualizations can also be provided such as bar charts)\n",
    "\n",
    "The code for making predictions on my final model is located in the Ipython notebook.\n",
    "\n",
    "Barchart showing output of top 5 probabilities for all 5 images is as follows:\n",
    "\n",
    "![alt text](output_images/softMax.jpg)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
