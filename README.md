# CUDA-impage-processing
Fabric defect detection using CUDA parallel processing

Fabric defect detection algorithms lack the real-time performance required in real applications because of their demanding computation. By using GPU, we are improving the processing speed of hardware system. This project will reduces the manual work required for testing the fabric defects. The GPU signicantly speeds up computation, especially achieving approximately av- erage thirty times than on the CPU.Our proposed work focuses on ecient real time implementation of an textile fabric defect detection algorithm called ITT using the concept of iterative tensor tracking on graphics processing unit(GPU).The algorithm adopts a new local image descriptor, spatial histograms of oriented gradients(S-HOG). To speedup the calculation required, ITT is implemented on the GPU using the Compute Unied Device Architecture programming model.

1Detailed Description of Methods

1.1CUDA Initialization:

CUDA initialization involves memory allocation for GPU device memory, Initial Data transfer from CPU to GPU memory. Since GPU cannot allocate its own memory, it needs to be allocated from CPU. Memory allocation on GPU is necessary to store the query. Simple Array data structure to store all computations result, which result into minimum number of memory cycles.
To perform all the operation we need to allocate memory, decide how much threads required for operation, and allocate specific work to each thread, after completing work free the memory space. All these operation can be accomplished using following:

1)	Allocate memory on both Device and Host using cudaMalloc () & malloc () respectively.

	CUDA programming model assumes a system composed of a host and a device, each with their own separate memory. Kernels can only operate out of device memory, so the runtime provides functions to allocate, deallocate, and copy device memory, as well as transfer data between host memory and device memory. Device memory can be allocated either as linear memory or as CUDA arrays. Linear memory is typically allocated using cudaMalloc () & Host memory can be allocated using malloc ().

	cudaMalloc(&device_variable, size);
	 malloc(size);	 

After allocating memory we will use it store intensity values of each pixel. And store these values on device memory for purpose of computation on Kernel/GPU. Then this memory also used to store all the result after completing computation because kernel function can’t return the values.

2)	Copying Content of memory using cudaMemcpy ()\

 The data transfer between host memory and device memory are typically done using cudaMemcpy ().Our System will use this for transferring intensity values from host to device memory, Gaussian mask from host to device memory, blurred values from device to host memory.
	cudaMemcpy(device_variable, host_variable, size, cudaMemcpyHostToDevice);
	cudaMemcpy(host_variable, device_variable, size, cudaMemcpyDeviceToHost);

3)	Deallocating memory using cudaFree ()

 After completing all the operation, free all the allocated memory. Free all the host variables & device variables.
	  cudaFree(device_variable); 
4)        Kernel & Work distribution to Thread
	C for CUDA extends C by allowing the programmer to define C functions, called kernels, that, when called, are executed N times in parallel by N different CUDA 
threads, as opposed to only once like regular C functions. A kernel is defined using the __global__ declaration specifier and the number of CUDA threads for each call is specified using a new <<<…>>> syntax: 

// Kernel definition 
__global__ void cal_Index (float* A, float* B) 
{ 
    ... 
} 
 
int main () 
{ 
    ... 
    // Kernel invocation 
   	cal_Index<<<BlocksPerGrid, threadsPerBlock>>>(A, B, ); 
} 

1.2 Each of the threads that execute a kernel is given a unique thread ID that is accessible within the kernel through the built-in threadIdx variable. We define threadsPerBlock, is the total number of threads in one block and BlocksPerGrid is the total number of blocks in a grid. We will mention this threadsPerBlock and BlocksPerGrid while calling Kernel functions. In Kernel function, we need to calculate index for copying elements from global memory to shared (local memory of GPU) memory as follows:

	index = threadIdx.x + (blockIdx.x * blockDim.x);


Implementation of Gabor Filter 

In this method there are three main steps selecting the Gabor filter from filter bank then applying the filter on image and then comparing the sample input and standard defect less image of fabric.

1)Selecting the Gabor filter
	In  input grayscale image (120x120 or 160x160) is densely filtered by a battery of Gabor filters at each scale and orientation. Therefore, at each pixel of the input image, filters of each size and orientation are centered.

2) Applying the filter :
		Before applying the Gabor function to the image convert it into grayscale image. For selected Gabor filter pass the values of  ƛ & σ as specified in the Gabor band table to the Gabor function. Apply this function  to the grayscale image and get the filtered output.

x_theta=image_resize(x,y)*cos(theta)+image_resize(x,y)*sin(theta);
y_theta=-image_resize(x,y)*sin(theta)+image_resize(x,y)*cos(theta);

	     gb(x,y)= exp(-(x_theta.^2/2*bw^2+ gamma^2*y_theta.^2/2*bw^2))
*cos(2*pi/lambda*x_theta+psi);

	   3) Monitoring  the defect:
		Take the output of sample image ,this filter image show defect in fabric with white colour pixels. If the filtered image contain the white pixels then the fabric is defected otherwise fabric is defect less.

2.1 Filtering
It is inevitable that all images taken from a camera will contain some amount of noise. To
prevent that noise is mistaken for edges, noise must be reduced. Therefore the image is first
smoothed by applying a Gaussian filter. The kernel of a Gaussian filter with a standard deviation
of _ = 1.4 is shown in Equation (1).

		2    4      5    4    2
		4    9      12   9    4
B= 1/159  *     5    12     15   12   5				(1)
		4    9      12   9    4
		2    4      5    4    2
2.2 Finding gradients
The Canny algorithm basically finds edges where the grayscale intensity of the image changes
the most. These areas are found by determining gradients of the image. Gradients at each pixel
in the smoothed image are determined by applying what is known as the Sobel-operator. First
step is to approximate the gradient in the x- and y-direction respectively by applying the kernels
shown in Equation (2).


			       -1    0    1
                     Kx  =     -2    0    2					
			       -1    0    1	
									(2)
			        1    2    1 
	             Ky  =	0    0    0					
			       -1   -2   -1
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

Contrast: Returns a measure of the intensity contrast between a pixel and its neighbor over the whole image.
 
 Range = [0 (size (GLCM, 1)-1)^2] 
 
Contrast is 0 for a constant image. Contrast is also known as variance and inertia.

Homogeneity: Returns a value that measures the closeness of the distribution of elements in the GLCM to the GLCM diagonal. 

Range = [0 1] Homogeneity is 1 for a diagonal GLCM.
