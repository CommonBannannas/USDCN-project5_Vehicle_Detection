## Writeup 
## Vehicle Detection
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_imgs/data_aug.png
[image2]: ./writeup_imgs/car_noncar.png
[image3]: ./writeup_imgs/hog.png
[image4]: ./writeup_imgs/window_search.png
[image5]: ./writeup_imgs/test_imgs.png
[image6]: ./writeup_imgs/heatmap.png
[image7]: ./writeup_imgs/thresholded.png
[image8]: ./writeup_imgs/labeled.png
[image9]: ./writeup_imgs/bboxes.png



## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

The project notebook: __Vehicle Detection.ipynb__ contains all the code and functions of the project.

The first step for my solution consisted of a quick exploration of the data set provided for training de SVC (Support Vector Classifier)
* Car images: 8792
* Not car images: 8968

![Car - Non Car][image2]

The dataset seems balanced, can we improve the accuracy a little bit by augmenting the non vehicle class?
I decided to give it a try and augment the data using the same process I used for project 2:

````
import scipy.ndimage
def create_variant(image):
   if (random.choice([1, 0])):
      image = scipy.ndimage.interpolation.shift(image, [random.randrange(-3, 3), random.randrange(-3, 3), 0])
    else:
       image = scipy.ndimage.interpolation.rotate(image, random.randrange(-8, 8), reshape=False)
   return image
    
````
I tilted and shifted randomly 2500 non car images. The result is some variants like theese ones:

![Data augmentation][image1]

The new dataset for training the SCV consisted of: 
* Car images: 8792
* Not car images: 11468

I decided to augment only the non car images to reinforce and reduce the number of false pisitives in the classification.
The acuracy of the SVC improved from:  0.982 to 0.9867 with the data augmentation.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Then I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

![HOG][image3]

After trying with different parameters I chose:
* svc train size: 16208
* svc test size: 4052
* orientations: 11
* pixels per cell: 16
* cells per block: 2
* feature vector size: 1188


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

````

svc = LinearSVC()
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print('svc train time:', round(t2-t, 2))
print('svc test accuracy: ', round(svc.score(X_test, y_test), 4))
t=time.time()
n_predict = 10
print('svc predicts: ', svc.predict(X_test[0:n_predict]))
print('for ',n_predict, 'labels: ', y_test[0:n_predict])

````


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to try different sizes and regions, the solution reference: [Jeremy Shannon's post](http://jeremyshannon.com/2017/03/17/udacity-sdcnd-vehicle-detection.html):

````
rectangles = []

ystart = 400
ystop = 464
scale = 1.0
rectangles.append(find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
orient, pix_per_cell, cell_per_block, None, None))

ystart = 416
ystop = 480
scale = 1.0
rectangles.append(find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, None, 
orient, pix_per_cell, cell_per_block, None, None))

ystart = 400
ystop = 496
scale = 1.5
rectangles.append(find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, None, 
orient, pix_per_cell, cell_per_block, None, None))

ystart = 432
ystop = 528
scale = 1.5
rectangles.append(find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, None, 
orient, pix_per_cell, cell_per_block, None, None))

ystart = 400
ystop = 528
scale = 2.0
rectangles.append(find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, None, 
orient, pix_per_cell, cell_per_block, None, None))

ystart = 432
ystop = 560
scale = 2.0
rectangles.append(find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, None, 
orient, pix_per_cell, cell_per_block, None, None))

ystart = 400
ystop = 596
scale = 3.5
rectangles.append(find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, None, 
orient, pix_per_cell, cell_per_block, None, None))

ystart = 464
ystop = 660
scale = 3.5
rectangles.append(find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, None, 
orient, pix_per_cell, cell_per_block, None, None))

rectangles = [item for sublist in rectangles for item in sublist] 
test_img_rects = draw_boxes(test_img, rectangles, color='random', thick=2)
plt.figure(figsize=(16,16))
plt.imshow(test_img_rects)

````
![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on the YUV c-space, with 11 orientations, 16 pixels per cells and 2 cells per block on all HOG channels.

![alt text][image5]

To optimize the performance of my classifier I augmented the non car data. (see above)

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used a class to store the 20 previous rectangles and keep the boxes from being too wobbly. Also I used a heatmap and thresholded the map to identify the vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


Here's an example results:



### Heatmaps:


* Heatmap:
![alt text][image6]



* Thresholded heatmap:
![alt text][image7]



* Labeled:
![alt text][image8]




### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image9]



---
