# CUDA-impage-processing
Fabric defect detection using CUDA parallel processing

Fabric defect detection algorithms lack the real-time performance required in real applications because of their demanding computation. By using GPU, we are improving the processing speed of hardware system. This project will reduces the manual work required for testing the fabric defects. The GPU signicantly speeds up computation, especially achieving approximately av- erage thirty times than on the CPU.Our proposed work focuses on ecient real time implementation of an textile fabric defect detection algorithm called ITT using the concept of iterative tensor tracking on graphics processing unit(GPU).The algorithm adopts a new local image descriptor, spatial histograms of oriented gradients(S-HOG). To speedup the calculation required, ITT is implemented on the GPU using the Compute Unied Device Architecture programming model.
Textile fabric flaw detection has always been an important part of product inspection and has been investigated in many literatures using vision sensors, image processing and pattern recognition techniques. Analysis of four different methods are done in this project. Their procedures and drawbacks are explained as follows.

a. Iterative Tensor Tracking using GPU
		The algorithm adopts a new local image descriptor, Spatial Histograms of Oriented Gradients (S-HOG). For a given textile fabric image, ITT iteratively updates and then analyzes S-HOG using tensor operations, in particular tensor decomposition to detect textile defects. The calculation of S-HOG uses a sliding window method. Each window represents a location where an area of 64 by 64 pixels is covered. The window is divided into four sub-blocks - 2 sub-blocks across and 2 sub-blocks down. The procedure of calculating HOG on the GPU follows the 2 steps 
          1.Calculate gradients of the given image,
          2. Compute histogram of gradient orientation of each cell and normalize it.
    Drawback of this method is while calculating the histograms of the image, there is a effect of shadow on image which gives drastic change in the histograms of image.
            
b. Image Difference
   This process adopts the pixel by pixel comparison of two images and shows the difference between two images but, due to noise it does not give appropriate results.

c. Gabor Filtering 
   Gabor filtering is the best method for texture analysis. Gabor filter contains two parts namely imaginary and real part. In fabric defect detection, we are focusing on only real part of Gabor filter. The Gabor bank contains different 64 filters according to the orientation and scale. For better results, it needs to select 16 orientations and 4 scales. Among all of above techniques this method is best because it gives appropriate results for any type of fabric.

d. Canny edge detection  
      The algorithm runs in 5 separate steps:
1. Smoothing: Blurring of the image to remove noise.
2. Finding gradients: The edges should be marked where the gradients of the image has large magnitudes.
3. Non-maximum suppression: Only local maxima should be marked as edges.
4. Double thresholding: Potential edges are determined by thresholding.
5. Edge tracking by hysteresis: Final edges are determined by suppressing all edges that are not connected to a very certain (strong) edge.

The Smoothing is done in order to prevent that noise in image which is mistaken for edges. Therefore the image is first smoothed by applying a Gaussian filter. In finding gradients the algorithm detects the edges where the grayscale intensity of the image changes the most. The Non-maxima suppression converts the “blurred” edges in the image of the gradient magnitudes to “sharp” edges. Basically this is done by preserving all local maxima in the gradient image, and deleting everything else. In Double thresholding Edge pixels stronger than the high threshold are marked as strong and get considered in process and rest are suppressed. Edge tracking can be implemented by BLOB-analysis (Binary Large Object). The edge pixels are divided into connected BLOB’s using 8-connected neighborhood. BLOB’s containing at least one strong edge pixel are then preserved, while other BLOB’s are suppressed.

For defect detection strategy, every image has a different noise so there is no such uniformity in fabric images. Further, while applying algorithms the thresholding values are changing for most images, so it’s difficult to set a threshold value resulting into rejection of this algorithm.


