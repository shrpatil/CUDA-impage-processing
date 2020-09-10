# CUDA-impage-processing
Fabric defect detection using CUDA parallel processing
Implementation
2.1 Filtering
It is inevitable that all images taken from a camera will contain some amount of noise. To
prevent that noise is mistaken for edges, noise must be reduced. Therefore the image is first
smoothed by applying a Gaussian filter. The kernel of a Gaussian filter with a standard deviation
of _ = 1.4 is shown in Equation (1).
2    4      5    4    2
4    9     12   9    4
                     B= 1/159  *      5    12   15  12   5				(1)
4    9     12   9    4
2    4      5    4    2


2.2 Finding gradients
The Canny algorithm basically finds edges where the grayscale intensity of the image changes
the most. These areas are found by determining gradients of the image. Gradients at each pixel
in the smoothed image are determined by applying what is known as the Sobel-operator. First
step is to approximate the gradient in the x- and y-direction respectively by applying the kernels
shown in Equation (2).


			       -1    0    1
                     Kx  = 	       -2     0     2					
			        -1    0     1	
									(2)
			         1    2     1 
	      Ky  =	         0    0     0					
			        -1    -2    -1
The gradient magnitudes (also known as the edge strengths) can then be determined as an
Euclidean distance measure by applying the law of Pythagoras as shown in Equation (3). It
is sometimes simplified by applying Manhattan distance measure as shown in Equation (4) to
reduce the computational complexity. The Euclidean distance measure has been applied to the
test image. 

		|G| = sqrt(  Gx2  +  Gy2   )				(3)
		|G| = |Gx|   +   |Gy|				(4)
where:
Gx and Gy are the gradients in the x- and y-directions respectively.

2.3 Non-maximum suppression
The purpose of this step is to convert the “blurred” edges in the image of the gradient magnitudes
to “sharp” edges. Basically this is done by preserving all local maxima in the gradient image,
and deleting everything else. The algorithm is for each pixel in the gradient image:
1. Round the gradient direction _ to nearest 45◦, corresponding to the use of an 8-connected
neighbourhood.
2. Compare the edge strength of the current pixel with the edge strength of the pixel in the
positive and negative gradient direction. I.e. if the gradient direction is north (theta =
90◦), compare with the pixels to the north and south.
3. If the edge strength of the current pixel is largest; preserve the value of the edge strength.
If not, suppress (i.e. remove) the value.


2.4 Double thresholding
The edge-pixels remaining after the non-maximum suppression step are (still) marked with their
strength pixel-by-pixel. Many of these will probably be true edges in the image, but some may
be caused by noise or color variations for instance due to rough surfaces. The simplest way to
discern between these would be to use a threshold, so that only edges stronger that a certain
value would be preserved. The Canny edge detection algorithm uses double thresholding. Edge
pixels stronger than the high threshold are marked as strong; edge pixels weaker than the low
threshold are suppressed and edge pixels between the two thresholds are marked as weak.

2.5 Edge tracking by hysteresis
Strong edges are interpreted as “certain edges”, and can immediately be included in the final
edge image. Weak edges are included if and only if they are connected to strong edges. Edge tracking can be implemented by BLOB-analysis (Binary Large OBject). The edge pixels
are divided into connected BLOB’s using 8-connected neighbourhood. BLOB’s containing at
least one strong edge pixel are then preserved, while other BLOB’s are suppressed.

2.6 Measure the Image Quality by using GLCM. 
The properties of glcm contain contrast and homogeneity.
Contrast: Returns a measure of the intensity contrast between a pixel and its neighbor over the whole             image.
 Range = [0 (size (GLCM, 1)-1)^2] 
Contrast is 0 for a constant image. Contrast is also known as variance and inertia. 
Homogeneity: Returns a value that measures the closeness of the distribution of elements in the GLCM to the GLCM diagonal. 
Range = [0 1] Homogeneity is 1 for a diagonal GLCM.
