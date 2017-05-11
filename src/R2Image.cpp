// Source file for image class



// Include files

#include "R2/R2.h"
#include "R2Pixel.h"
#include "R2Image.h"
#include "svd.h"
#include "math.h"

////////////////////////////////////////////////////////////////////////
// Constructors/Destructors
////////////////////////////////////////////////////////////////////////


R2Image::
R2Image(void)
  : pixels(NULL),
	npixels(0),
	width(0),
	height(0)
{
}



R2Image::
R2Image(const char *filename)
  : pixels(NULL),
	npixels(0),
	width(0),
	height(0)
{
  // Read image
  Read(filename);
}



R2Image::
R2Image(int width, int height)
  : pixels(NULL),
	npixels(width * height),
	width(width),
	height(height)
{
  // Allocate pixels
  pixels = new R2Pixel [ npixels ];
  assert(pixels);
}



R2Image::
R2Image(int width, int height, const R2Pixel *p)
  : pixels(NULL),
	npixels(width * height),
	width(width),
	height(height)
{
  // Allocate pixels
  pixels = new R2Pixel [ npixels ];
  assert(pixels);

  // Copy pixels
  for (int i = 0; i < npixels; i++)
	pixels[i] = p[i];
}



R2Image::
R2Image(const R2Image& image)
  : pixels(NULL),
	npixels(image.npixels),
	width(image.width),
	height(image.height)

{
  // Allocate pixels
  pixels = new R2Pixel [ npixels ];
  assert(pixels);

  // Copy pixels
  for (int i = 0; i < npixels; i++)
	pixels[i] = image.pixels[i];
}



R2Image::
~R2Image(void)
{
  // Free image pixels
  if (pixels) delete [] pixels;
}



R2Image& R2Image::
operator=(const R2Image& image)
{
  // Delete previous pixels
  if (pixels) { delete [] pixels; pixels = NULL; }

  // Reset width and height
  npixels = image.npixels;
  width = image.width;
  height = image.height;

  // Allocate new pixels
  pixels = new R2Pixel [ npixels ];
  assert(pixels);

  // Copy pixels
  for (int i = 0; i < npixels; i++)
	pixels[i] = image.pixels[i];

  // Return image
  return *this;
}


void R2Image::
svdTest(void)
{
	// fit a 2D conic to five points
	R2Point p1(1.2,3.5);
	R2Point p2(2.1,2.2);
	R2Point p3(0.2,1.6);
	R2Point p4(0.0,0.5);
	R2Point p5(-0.2,4.2);

	// build the 5x6 matrix of equations
	double** linEquations = dmatrix(1,5,1,6);

	linEquations[1][1] = p1[0]*p1[0];
	linEquations[1][2] = p1[0]*p1[1];
	linEquations[1][3] = p1[1]*p1[1];
	linEquations[1][4] = p1[0];
	linEquations[1][5] = p1[1];
	linEquations[1][6] = 1.0;

	linEquations[2][1] = p2[0]*p2[0];
	linEquations[2][2] = p2[0]*p2[1];
	linEquations[2][3] = p2[1]*p2[1];
	linEquations[2][4] = p2[0];
	linEquations[2][5] = p2[1];
	linEquations[2][6] = 1.0;

	linEquations[3][1] = p3[0]*p3[0];
	linEquations[3][2] = p3[0]*p3[1];
	linEquations[3][3] = p3[1]*p3[1];
	linEquations[3][4] = p3[0];
	linEquations[3][5] = p3[1];
	linEquations[3][6] = 1.0;

	linEquations[4][1] = p4[0]*p4[0];
	linEquations[4][2] = p4[0]*p4[1];
	linEquations[4][3] = p4[1]*p4[1];
	linEquations[4][4] = p4[0];
	linEquations[4][5] = p4[1];
	linEquations[4][6] = 1.0;

	linEquations[5][1] = p5[0]*p5[0];
	linEquations[5][2] = p5[0]*p5[1];
	linEquations[5][3] = p5[1]*p5[1];
	linEquations[5][4] = p5[0];
	linEquations[5][5] = p5[1];
	linEquations[5][6] = 1.0;

	printf("\n Fitting a conic to five points:\n");
	printf("Point #1: %f,%f\n",p1[0],p1[1]);
	printf("Point #2: %f,%f\n",p2[0],p2[1]);
	printf("Point #3: %f,%f\n",p3[0],p3[1]);
	printf("Point #4: %f,%f\n",p4[0],p4[1]);
	printf("Point #5: %f,%f\n",p5[0],p5[1]);

	// compute the SVD
	double singularValues[7]; // 1..6
	double** nullspaceMatrix = dmatrix(1,6,1,6);
	svdcmp(linEquations, 5, 6, singularValues, nullspaceMatrix);

	// get the result
	printf("\n Singular values: %f, %f, %f, %f, %f, %f\n",singularValues[1],singularValues[2],singularValues[3],singularValues[4],singularValues[5],singularValues[6]);

	// find the smallest singular value:
	int smallestIndex = 1;
	for(int i=2;i<7;i++) if(singularValues[i]<singularValues[smallestIndex]) smallestIndex=i;

	// solution is the nullspace of the matrix, which is the column in V corresponding to the smallest singular value (which should be 0)
	printf("Conic coefficients: %f, %f, %f, %f, %f, %f\n",nullspaceMatrix[1][smallestIndex],nullspaceMatrix[2][smallestIndex],nullspaceMatrix[3][smallestIndex],nullspaceMatrix[4][smallestIndex],nullspaceMatrix[5][smallestIndex],nullspaceMatrix[6][smallestIndex]);

	// make sure the solution is correct:
	printf("Equation #1 result: %f\n",	p1[0]*p1[0]*nullspaceMatrix[1][smallestIndex] +
										p1[0]*p1[1]*nullspaceMatrix[2][smallestIndex] +
										p1[1]*p1[1]*nullspaceMatrix[3][smallestIndex] +
										p1[0]*nullspaceMatrix[4][smallestIndex] +
										p1[1]*nullspaceMatrix[5][smallestIndex] +
										nullspaceMatrix[6][smallestIndex]);

	printf("Equation #2 result: %f\n",	p2[0]*p2[0]*nullspaceMatrix[1][smallestIndex] +
										p2[0]*p2[1]*nullspaceMatrix[2][smallestIndex] +
										p2[1]*p2[1]*nullspaceMatrix[3][smallestIndex] +
										p2[0]*nullspaceMatrix[4][smallestIndex] +
										p2[1]*nullspaceMatrix[5][smallestIndex] +
										nullspaceMatrix[6][smallestIndex]);

	printf("Equation #3 result: %f\n",	p3[0]*p3[0]*nullspaceMatrix[1][smallestIndex] +
										p3[0]*p3[1]*nullspaceMatrix[2][smallestIndex] +
										p3[1]*p3[1]*nullspaceMatrix[3][smallestIndex] +
										p3[0]*nullspaceMatrix[4][smallestIndex] +
										p3[1]*nullspaceMatrix[5][smallestIndex] +
										nullspaceMatrix[6][smallestIndex]);

	printf("Equation #4 result: %f\n",	p4[0]*p4[0]*nullspaceMatrix[1][smallestIndex] +
										p4[0]*p4[1]*nullspaceMatrix[2][smallestIndex] +
										p4[1]*p4[1]*nullspaceMatrix[3][smallestIndex] +
										p4[0]*nullspaceMatrix[4][smallestIndex] +
										p4[1]*nullspaceMatrix[5][smallestIndex] +
										nullspaceMatrix[6][smallestIndex]);

	printf("Equation #5 result: %f\n",	p5[0]*p5[0]*nullspaceMatrix[1][smallestIndex] +
										p5[0]*p5[1]*nullspaceMatrix[2][smallestIndex] +
										p5[1]*p5[1]*nullspaceMatrix[3][smallestIndex] +
										p5[0]*nullspaceMatrix[4][smallestIndex] +
										p5[1]*nullspaceMatrix[5][smallestIndex] +
										nullspaceMatrix[6][smallestIndex]);

	R2Point test_point(0.34,-2.8);

	printf("A point off the conic: %f\n",	test_point[0]*test_point[0]*nullspaceMatrix[1][smallestIndex] +
											test_point[0]*test_point[1]*nullspaceMatrix[2][smallestIndex] +
											test_point[1]*test_point[1]*nullspaceMatrix[3][smallestIndex] +
											test_point[0]*nullspaceMatrix[4][smallestIndex] +
											test_point[1]*nullspaceMatrix[5][smallestIndex] +
											nullspaceMatrix[6][smallestIndex]);

	return;
}

////////////////////////////////////////////////////////////////////////
// Image processing functions
// YOU IMPLEMENT THE FUNCTIONS IN THIS SECTION
////////////////////////////////////////////////////////////////////////

// Per-pixel Operations ////////////////////////////////////////////////

void R2Image::
Brighten(double factor)
{
  // Brighten the image by multiplying each pixel component by the factor.
  // This is implemented for you as an example of how to access and set pixels
  for (int i = 0; i < width; i++) {
	for (int j = 0;  j < height; j++) {
	  Pixel(i,j) *= factor;
	  Pixel(i,j).Clamp();
	}
  }
}

void R2Image::
SobelX(void)
{
	// Apply the Sobel oprator to the image in X direction

  R2Image originalImage(*this);

  for(int y=0; y<originalImage.height; y++) {
	for(int x=0; x<originalImage.width; x++) {


	  // Edge case handling
	  int xMinus1 = x-1;
	  int xPlus1 = x+1;
	  int yMinus1 = y-1;
	  int yPlus1 = y+1;

	  if(x < 1) {
		xMinus1 = x;
	  }
	  if(y < 1) {
		yMinus1 = y;
	  }
	  if(x == this->width-1) {
		xPlus1 = x;
	  }
	  if(y == this->height-1) {
		yPlus1 = y;
	  }
	  //

	   R2Pixel tempPixel = -1 * (originalImage.Pixel(xMinus1,yMinus1))
					+1 * (originalImage.Pixel(xPlus1,yMinus1))
					-2 * (originalImage.Pixel(xMinus1,y))
					+2 * (originalImage.Pixel(xPlus1,y))
					-1 * (originalImage.Pixel(xMinus1,yPlus1))
					+1 * (originalImage.Pixel(xPlus1,yPlus1));
		//tempPixel.Clamp();

	  this->SetPixel(x, y, tempPixel);

	  //R2Pixel(rTemp, gTemp, bTemp, 1.0)
	}
  }
}

void R2Image::
SobelY(void)
{
	// Apply the Sobel oprator to the image in Y direction

  R2Image originalImage(*this);

  for(int y=0; y<originalImage.height; y++) {
	for(int x=0; x<originalImage.width; x++) {

	  // Edge case handling
	  int xMinus1 = x-1;
	  int xPlus1 = x+1;
	  int yMinus1 = y-1;
	  int yPlus1 = y+1;

	  if(x < 1) {
		xMinus1 = x;
	  }
	  if(y < 1) {
		yMinus1 = y;
	  }
	  if(x == this->width-1) {
		xPlus1 = x;
	  }
	  if(y == this->height-1) {
		yPlus1 = y;
	  }
	  //

	R2Pixel tempPixel = -1 * (originalImage.Pixel(xMinus1,yMinus1))
					-2 * (originalImage.Pixel(x,yMinus1))
					-1 * (originalImage.Pixel(xPlus1,yMinus1))
					+1 * (originalImage.Pixel(xMinus1,yPlus1))
					+2 * (originalImage.Pixel(x,yPlus1))
					+1 * (originalImage.Pixel(xPlus1,yPlus1));

	//tempPixel.Clamp();

	  this->SetPixel(x, y, tempPixel);
	}
  }
}


double** homographyEstimation(double** p1, double** p2) {

  double** A = dmatrix(1,8,1,9);

  for(int i=0; i<4; i++) {
    int x = p1[i][0];
    int y = p1[i][1];
    int u = p2[i][0];
    int v = p2[i][1];

    double upper[9] = {x, y, 1, 0, 0, 0, -u*x, -u*y, -u};
    double lower[9] = {0, 0, 0, x, y, 1, -v*x, -v*y, -v};


    // loop through to add into A array
    for(int j=0; j<9; j++) {
      A[2*i+1][j+1] = upper[j];
      A[2*i+2][j+1] = lower[j];
    }
  }

  double singularValues[9];
  double** nullspaceMatrix = dmatrix(1,9,1,9);

  svdcmp(A, 8, 9, singularValues, nullspaceMatrix);

  // find the smallest singular value
	int smallestIndex = 1;
	for(int i=2;i<10;i++) {
    if(singularValues[i]<singularValues[smallestIndex]) {
      smallestIndex=i;
    }
  }

  double H[9];
  for(int i=0; i<9; i++) {
    H[i] = nullspaceMatrix[i+1][smallestIndex];
  }


	// solution is the nullspace of the matrix, which is the column in V corresponding to the smallest singular value (which should be 0)
	//printf("H matrix values: %f, %f, %f, %f, %f, %f, %f, %f, %f\n",H[0],H[1],H[2],H[3],H[4],H[5],H[6],H[7],H[8]);

  double** H_matrix = dmatrix(0,3,0,3);
  for(int i=0; i<9; i++) {
    H_matrix[i/3][i%3] = H[i];
  }

  return H_matrix;
}


void R2Image::
LoG(void)
{
  // Apply the LoG oprator to the image


  // FILL IN IMPLEMENTATION HERE (REMOVE PRINT STATEMENT WHEN DONE)
  //fprintf(stderr, "LoG() not implemented\n");
}



void R2Image::
ChangeSaturation(double factor)
{
  // Changes the saturation of an image
  // Find a formula that changes the saturation without affecting the image brightness

  // FILL IN IMPLEMENTATION HERE (REMOVE PRINT STATEMENT WHEN DONE)
  fprintf(stderr, "ChangeSaturation(%g) not implemented\n", factor);
}





// Linear filtering ////////////////////////////////////////////////

int R2Image::
pixelEdgeCase(int index, int maxVal) {
	if(index < 0) {
		return 0;
	}
	else if(index >= maxVal) {
		return maxVal-1;
	}
	else return index;
}

// calculates the resulting value from the gaussian function (from Wikipedia)
// given an index (offset from origin pixel) and sigma value
double R2Image::
gaussianFunc(int index, double sigma) {

	double ePower = -1 * (pow(index, 2)) / (2 * pow(sigma, 2));
	double normCoeff = 1 / (sqrt(2 * 3.14159) * sigma);
	double result = normCoeff * pow(2.7182818, ePower);

	return result;
}

// creates a vector with values for the 1-D Gaussian kernel
std::vector<double> R2Image::
generateGaussianKernel(double sigma) {
	int sigmaInt = (int)sigma; // sigma value truncated to an Int, for creating vec length
  int kernelWidth = 6 * sigmaInt + 1; // full length of vec for the kernel values

	std::vector<double> kernel(kernelWidth);

	for(int i=0; i<kernelWidth; i++) {
		kernel.at(i) = this->gaussianFunc(abs(i-(3*sigmaInt)), sigma);
		//printf("Kernel value %i: %f\n", i, kernel[i]);
	}

	return kernel;
}

void R2Image::
Blur(double sigma)
{
  // Gaussian blur of the image. Separable solution is preferred

	std::vector<double> gaussKern = generateGaussianKernel(sigma);

/*
	for(int i=0; i<kernelWidth; i++) {
		printf("Kernel value %i: %f\n", i, gaussKern.at(i));
	}
  */

  	R2Image originalImage(*this);

	for(int y=0; y<originalImage.height; y++) {
		for(int x=0; x<originalImage.width; x++) {

			R2Pixel tempPixel;
			for(int i=(-3*(int)sigma); i<(3*(int)sigma); i++) {
				tempPixel = tempPixel + gaussKern.at(i+(3*(int)sigma)) * (originalImage.Pixel(pixelEdgeCase(x+i, originalImage.width), y));
			}

			//tempPixel.Clamp();
			SetPixel(x, y, tempPixel);
		}
	}

	R2Image horizontallyBlurred(*this);

	for(int y=0; y<horizontallyBlurred.height; y++) {
		for(int x=0; x<horizontallyBlurred.width; x++) {

			R2Pixel tempPixel;
			for(int i=(-3*(int)sigma); i<(3*(int)sigma); i++) {
				tempPixel = tempPixel + gaussKern.at(i+(3*(int)sigma)) * (horizontallyBlurred.Pixel(x, pixelEdgeCase(y+i, horizontallyBlurred.height)));
			}

			//tempPixel.Clamp();
			SetPixel(x, y, tempPixel);
		}
	}
}

bool sortByPixelStrength(ContextPixel lhs, ContextPixel rhs) {
  return lhs.pixel.Luminance() > rhs.pixel.Luminance();
}

bool distanceIsGreaterThan(ContextPixel lhs, ContextPixel rhs, double minDistance) {
  double distance = sqrt(pow((lhs.xCoord - rhs.xCoord), 2) + pow((lhs.yCoord - rhs.yCoord), 2));
  return distance > minDistance;
}

void R2Image::
Harris(double sigma)
{
  // Harris corner detector. Make use of the previously developed filters, such as the Gaussian blur filter
	// Output should be 50% grey at flat regions, white at corners and black/dark near edges

  // create sobel images (x and y)
  R2Image sobelXImg(*this);
  R2Image sobelYImg(*this);
  sobelXImg.SobelX();
  sobelYImg.SobelY();

  // create Ix, Iy, Ixy
  R2Image ix_squared(width, height);
  R2Image iy_squared(width, height);
  R2Image ixy_img(width, height);

  // square Ix and Iy, multiply together into Ixy
  for(int x=0; x<width; x++) {
    for(int y=0; y<height; y++) {
      ix_squared.SetPixel(x, y, (sobelXImg.Pixel(x,y) * sobelXImg.Pixel(x,y)));
      iy_squared.SetPixel(x, y, (sobelYImg.Pixel(x,y) * sobelYImg.Pixel(x,y)));
      ixy_img.SetPixel(x, y, (sobelXImg.Pixel(x,y) * sobelYImg.Pixel(x,y)));
    }
  }

  // blur temp images based on sigma argument
  ix_squared.Blur(sigma);
  iy_squared.Blur(sigma);
  ixy_img.Blur(sigma);

  R2Pixel offsetPix(0.5, 0.5, 0.5, 1);

  for(int x=0; x<width; x++) {
    for(int y=0; y<height; y++) {

      R2Pixel tempPixel = ix_squared.Pixel(x,y)*iy_squared.Pixel(x,y) - ixy_img.Pixel(x,y)*ixy_img.Pixel(x,y) - 0.04*((ix_squared.Pixel(x,y) + iy_squared.Pixel(x,y))*(ix_squared.Pixel(x,y) + iy_squared.Pixel(x,y)));
      tempPixel += offsetPix;

      tempPixel.Clamp();

      SetPixel(x, y, tempPixel);
    }
  }
}

void R2Image::
Sharpen()
{
  // Sharpen an image using a linear filter. Use a kernel of your choosing.
  R2Image originalImage(*this);

  for(int y=0; y<originalImage.height; y++) {
	for(int x=0; x<originalImage.width; x++) {

	  // Edge case handling
	  int xMinus1 = x-1;
	  int xPlus1 = x+1;
	  int yMinus1 = y-1;
	  int yPlus1 = y+1;

	  if(x == 0) {
		xMinus1 = x;
	  }
	  if(y == 0) {
		yMinus1 = y;
	  }
	  if(x == this->width-1) {
		xPlus1 = x;
	  }
	  if(y == this->height-1) {
		yPlus1 = y;
	  }

	  R2Pixel tempPixel = -1 * (originalImage.Pixel(x,yMinus1))
					      -1 * (originalImage.Pixel(xMinus1,y))
					      +5 * (originalImage.Pixel(x,y))
					      -1 * (originalImage.Pixel(xPlus1,y))
					      -1 * (originalImage.Pixel(x,yPlus1));

	  tempPixel.Clamp();

	  this->SetPixel(x, y, tempPixel);
	}
  }
}

void R2Image::
line(int x0, int x1, int y0, int y1, float r, float g, float b)
{
  if(x0>x1)
	{
		int x=y1;
		y1=y0;
		y0=x;

		x=x1;
		x1=x0;
		x0=x;
	}

     int deltax = x1 - x0;
     int deltay = y1 - y0;
     float error = 0;
     float deltaerr = 0.0;
	 if(deltax!=0) deltaerr =fabs(float(float(deltay) / deltax));    // Assume deltax != 0 (line is not vertical),
           // note that this division needs to be done in a way that preserves the fractional part
     int y = y0;
     for(int x=x0;x<=x1;x++)
	 {
		 Pixel(x,y).Reset(r,g,b,1.0);
         error = error + deltaerr;
         if(error>=0.5)
		 {
			 if(deltay>0) y = y + 1;
			 else y = y - 1;

             error = error - 1.0;
		 }
	 }
	 if(x0>3 && x0<width-3 && y0>3 && y0<height-3)
	 {
		 for(int x=x0-3;x<=x0+3;x++)
		 {
			 for(int y=y0-3;y<=y0+3;y++)
			 {
				 Pixel(x,y).Reset(r,g,b,1.0);
			 }
		 }
	 }
}

typedef struct {
  R2Pixel pixel;
  int xCoord;
  int yCoord;
  double score;
} SSD_pixel;

/*
bool sortBySSDScore(SSD_pixel lhs, SSD_pixel rhs) {
  return lhs.score < rhs.score;
}
*/

double SSD_score(R2Image orig, R2Image comp, int x1, int y1, int x2, int y2) {
  int featureSize = 5;
  double score = 0;
  // sums squared difference of pixels for featureSize^2 pixel area
  for(int x=(-1*featureSize); x<featureSize; x++) {
    for(int y=(-1*featureSize); y<featureSize; y++) {
      //printf("[SSD_score] finding difference between pixel1 (%d, %d) and pixel2 (%d, %d)\n", x1+x, y1+y, x2+x, y2+y);
      double diff = orig.Pixel(x1+x,y1+y).Luminance() - comp.Pixel(x2+x,y2+y).Luminance();
      score += diff*diff;
    }
  }
  //printf("  score: %f\n", score);
  return score;
}

bool sortByNumSupporters(TranslationVector lhs, TranslationVector rhs) {
  return lhs.supporters > rhs.supporters;
}

std::vector<ContextPixel> R2Image::
findBestFeatures() {
  printf("Finding best features\n");

  R2Image originalImageHarris(*this);

  originalImageHarris.Harris(2.0);

  // create vector of all pixels
  std::vector<ContextPixel> sortedPixels(width * height);

  for(int x=0; x<width; x++) {
    for(int y=0; y<height; y++) {
      ContextPixel tempPix;
      tempPix.pixel = originalImageHarris.Pixel(x,y);
      tempPix.xCoord = x;
      tempPix.yCoord = y;
      sortedPixels.push_back(tempPix);
    }
  }

  // sort vector by pixel sortByPixelStrength
  std::sort(sortedPixels.begin(), sortedPixels.end(), sortByPixelStrength);

  //R2Pixel redPixel = R2Pixel(1.0, 0.0, 0.0, 1.0);

  int numFeat = 150;
  int featCount = 0;

  std::vector<ContextPixel> foundFeatures(numFeat);

  int index = 0;
  while(featCount < numFeat) {
    ContextPixel currentPixel = sortedPixels.at(index);
    //printf(" index: %d x: %d y: %d\n", index, currentPixel.xCoord, currentPixel.yCoord);

    // if the current pixel is more than 10 pixels away from any already-found feature
    bool pixelIsValid = true;
    for(int i=0; i<foundFeatures.size(); i++) {
      if(!distanceIsGreaterThan(currentPixel, foundFeatures.at(i), 10.0)) {
        pixelIsValid = false;
      }
    }

    // draw boxes around highest-scoring pixels from highest to lowest, add said pixel to foundFeatures vector
    if(pixelIsValid) {
      printf("added a valid pixel index: %d x: %d y: %d\n", index, currentPixel.xCoord, currentPixel.yCoord);
      foundFeatures.at(featCount) = currentPixel;
      featCount++;
    }
    index++;
  }

  return foundFeatures;
}

std::vector<ContextPixel> R2Image::
blendOtherImageTranslated(R2Image * otherImage, std::vector<ContextPixel> foundFeatures)
{
	// find at least 100 features on this image, and another 100 on the "otherImage". Based on these,
	// compute the matching translation (pixel precision is OK), and blend the translated "otherImage"
	// into this image with a 50% opacity.

  printf("matching features\n");

  int searchWidth = width * 0.2;
  int searchHeight = height * 0.2;
  int featureSize = 5;

  std::vector<ContextPixel> matchedFeatures(foundFeatures.size());

  // Find Matches using SSD
  for(int i=0; i<foundFeatures.size(); i++) {
    ContextPixel currentPixel = foundFeatures.at(i);
    //printf("Matching feature pixel #%d, x: %d y: %d luminance: %f\n", i, currentPixel.xCoord, currentPixel.yCoord, currentPixel.pixel.Luminance());

    int curX = currentPixel.xCoord;
    int curY = currentPixel.yCoord;
    SSD_pixel bestMatch;
    bestMatch.score = 9999999;

    // looping within search area to find a matched feature
    for(int x=(-1*searchWidth/2); x<(searchWidth/2); x++) {
      for(int y=(-1*searchHeight/2); y<(searchHeight/2); y++) {
        // for each pixel in the search window, calculate the SSD
        // score for the feature area around it
        int adjX = curX + x;
        int adjY = curY + y;

        // SSD scoring procedure
        double score = 0;
        // sums squared difference of pixels for featureSize^2 pixel area
        for(int a=(-1*featureSize); a<featureSize; a++) {
          for(int b=(-1*featureSize); b<featureSize; b++) {
            double diff = Pixel(pixelEdgeCase(curX+a, width), pixelEdgeCase(curY+b, height)).Luminance() - otherImage->Pixel(pixelEdgeCase(adjX+a, width), pixelEdgeCase(adjY+b, height)).Luminance();
            score += diff*diff;
          }
        }
        if(score < bestMatch.score) {
          bestMatch.xCoord = adjX;
          bestMatch.yCoord = adjY;
          bestMatch.score = score;
        }
      }
    }
    ContextPixel newMatch;

    newMatch.xCoord = bestMatch.xCoord;
    newMatch.yCoord = bestMatch.yCoord;

    matchedFeatures.at(i) = newMatch;
    printf("Matched pixel (%d %d) with translated img pixel (%d %d)\n", curX, curY, newMatch.xCoord, newMatch.yCoord);
  }

  return matchedFeatures;
}

TranslationVector R2Image::
vectorRANSAC(std::vector<ContextPixel> before, std::vector<ContextPixel> after) {
    // RANSAC

    // loop through all translation vectors (or limit, can just pick 100 times?)
      // check all other vectors against it

      // but all you should do is compute for eachthe difference vector between
      // those, and find the length of that difference vector to understand how
      // "good" it is

      // count how many "supporters" it has, i.e. vectors that are within a
      // threshold

    // pick the vector that has the most supporters, and then mark the
    // ones within the threshold as "good" matches

    // or, take the average of all those matching vectors to mark outliers
    //=====================================================================

    // Add translation vectors to array
    std::vector<TranslationVector> translationVectors(after.size());

    for(int i=0; i<after.size(); i++) {
      TranslationVector vec;
      vec.x1 = before.at(i).xCoord;
      vec.y1 = before.at(i).yCoord;
      vec.x2 = after.at(i).xCoord;
      vec.y2 = after.at(i).yCoord;
      vec.x = vec.x2 - vec.x1;
      vec.y = vec.y2 - vec.y1;
      //printf("translationVector %d x: %d y: %d\n", i, vec.x, vec.y);
      translationVectors.push_back(vec);
    }

    // compensate for bug that causes 150 error values at beginning of array??
    translationVectors.erase(translationVectors.begin(), translationVectors.begin()+150);

    // RANSAC
    double acceptThresh = 5.0;

    // score each vector based on similarity to other vectors
    for(int i=0; i<translationVectors.size(); i++) {
      TranslationVector vec1 = translationVectors.at(i);
      vec1.supporters = 0;
      for(int j=0; j<translationVectors.size(); j++) {
        TranslationVector vec2 = translationVectors.at(j);
        int xDiff = vec2.x - vec1.x;
        int yDiff = vec2.y - vec1.y;
        double diffLength = sqrt(xDiff*xDiff + yDiff*yDiff); // dist formula
        // if the length of the difference vector is less than the threshold
        //printf("xDiff: %d yDiff: %d diffLength: %f\n", xDiff, yDiff, diffLength);
        if(diffLength < acceptThresh) {
          // increment num of supporters
          vec1.supporters++;
        }
      }
      vec1.supporters -= 150;
      //printf("vector %d has %d supporters\n", i, vec1.supporters);
    }

    std::sort(translationVectors.begin(), translationVectors.end(), sortByNumSupporters);
    TranslationVector winner = translationVectors.at(0);


    // draw vectors in different colors based on status
    for(int i=0; i<translationVectors.size(); i++) {
      TranslationVector comp = translationVectors.at(i);
      int xDiff = comp.x - winner.x;
      int yDiff = comp.y - winner.y;
      double diffLength = sqrt(xDiff*xDiff + yDiff*yDiff); // dist form
      // if the length of the difference vector is less than the threshold
      comp.outlier = diffLength > acceptThresh;
      if(comp.outlier) {
        line(comp.x1, comp.x2, comp.y1, comp.y2, 1.0, 0.0, 0.0);
      }
      else {
        line(comp.x1, comp.x2, comp.y1, comp.y2, 0.0, 1.0, 0.0);
      }
    }
    
    return winner;
}

double* matrixMult(double** mat, double* vec) {
  double* resultVec = dvector(0,3);
  for(int i=0; i<3; i++) {
    resultVec[i] = mat[i][0]*vec[0] + mat[i][1]*vec[1] + mat[i][2]*vec[2];
  }
  //printf("%f %f %f\n", resultVec[0], resultVec[1], resultVec[2]);
  return resultVec;
}

typedef struct {
  double** homographyMatrix;
  int supporters;
} HomographyMat;

bool sortHomogByNumSupporters(HomographyMat lhs, HomographyMat rhs) {
  return lhs.supporters > rhs.supporters;
}

void R2Image::
blendOtherImageHomography(R2Image * otherImage)
{

  // find at least 100 features on this image, and another 100 on the "otherImage". Based on these,
	// compute the matching translation (pixel precision is OK), and blend the translated "otherImage"
	// into this image with a 50% opacity.

  R2Image originalImageHarris(*this);

  originalImageHarris.Harris(2.0);

  // create vector of all pixels
  std::vector<ContextPixel> sortedPixels(width * height);

  for(int x=0; x<width; x++) {
    for(int y=0; y<height; y++) {
      ContextPixel tempPix;
      tempPix.pixel = originalImageHarris.Pixel(x,y);
      tempPix.xCoord = x;
      tempPix.yCoord = y;
      sortedPixels.push_back(tempPix);
    }
  }

  // sort vector by pixel sortByPixelStrength
  std::sort(sortedPixels.begin(), sortedPixels.end(), sortByPixelStrength);

  //R2Pixel redPixel = R2Pixel(1.0, 0.0, 0.0, 1.0);

  int numFeat = 150;
  int featCount = 0;

  std::vector<ContextPixel> foundFeatures(numFeat);

  int index = 0;
  while(featCount < numFeat) {
    ContextPixel currentPixel = sortedPixels.at(index);
    //printf(" index: %d x: %d y: %d\n", index, currentPixel.xCoord, currentPixel.yCoord);

    // if the current pixel is more than 10 pixels away from any already-found feature
    bool pixelIsValid = true;
    for(int i=0; i<foundFeatures.size(); i++) {
      if(!distanceIsGreaterThan(currentPixel, foundFeatures.at(i), 10.0)) {
        pixelIsValid = false;
      }
    }

    // draw boxes around highest-scoring pixels from highest to lowest, add said pixel to foundFeatures vector
    if(pixelIsValid) {
      //printf("added a valid pixel index: %d x: %d y: %d\n", index, currentPixel.xCoord, currentPixel.yCoord);
      foundFeatures.at(featCount) = currentPixel;
      featCount++;
    }
    index++;
  }

  int searchWidth = width * 0.2;
  int searchHeight = height * 0.2;
  int featureSize = 5;


  std::vector<ContextPixel> matchedFeatures(numFeat);

  // Find Matches using SSD
  for(int i=0; i<foundFeatures.size(); i++) {
    ContextPixel currentPixel = foundFeatures.at(i);
    //printf("Matching feature pixel #%d, x: %d y: %d luminance: %f\n", i, currentPixel.xCoord, currentPixel.yCoord, currentPixel.pixel.Luminance());

    int curX = currentPixel.xCoord;
    int curY = currentPixel.yCoord;
    SSD_pixel bestMatch;
    bestMatch.score = 9999999;

    // looping within search area to find a matched feature
    for(int x=(-1*searchWidth/2); x<(searchWidth/2); x++) {
      for(int y=(-1*searchHeight/2); y<(searchHeight/2); y++) {
        // for each pixel in the search window, calculate the SSD
        // score for the feature area around it
        int adjX = curX + x;
        int adjY = curY + y;

        // SSD scoring procedure
        double score = 0;
        // sums squared difference of pixels for featureSize^2 pixel area
        for(int a=(-1*featureSize); a<featureSize; a++) {
          for(int b=(-1*featureSize); b<featureSize; b++) {
            double diff = Pixel(pixelEdgeCase(curX+a, width), pixelEdgeCase(curY+b, height)).Luminance() - otherImage->Pixel(pixelEdgeCase(adjX+a, width), pixelEdgeCase(adjY+b, height)).Luminance();
            score += diff*diff;
          }
        }
        if(score < bestMatch.score) {
          bestMatch.xCoord = adjX;
          bestMatch.yCoord = adjY;
          bestMatch.score = score;
        }
      }
    }
    ContextPixel newMatch;

    newMatch.xCoord = bestMatch.xCoord;
    newMatch.yCoord = bestMatch.yCoord;

    matchedFeatures.at(i) = newMatch;
  }

  // RANSAC

  // Add translation vectors to array
  std::vector<TranslationVector> translationVectors(numFeat);
  for(int i=0; i<numFeat; i++) {
    TranslationVector vec;
    vec.x1 = foundFeatures.at(i).xCoord;
    vec.y1 = foundFeatures.at(i).yCoord;
    vec.x2 = matchedFeatures.at(i).xCoord;
    vec.y2 = matchedFeatures.at(i).yCoord;
    vec.x = vec.x2 - vec.x1;
    vec.y = vec.y2 - vec.y1;
    translationVectors.push_back(vec);
  }
  // compensate for bug that causes 150 error values at beginning of array??
  translationVectors.erase(translationVectors.begin(), translationVectors.begin()+150);


  int numTrials = 500;
  double acceptThresh = 5.0;

  std::vector<HomographyMat> homogMatricies(numTrials);

  for(int i=0; i<numTrials; i++) {

    // create arrays to hold 4 random points
    double** before = dmatrix(0,4,0,2);
    double** after = dmatrix(0,4,0,2);

    // pick 4 random points and add to vector
    for(int j=0; j<4; j++) {
      int randomNum = rand() % numFeat;
      TranslationVector vec = translationVectors.at(randomNum);

      before[j][0] = vec.x1;
      before[j][1] = vec.y1;
      after[j][0] = vec.x2;
      after[j][1] = vec.y2;
    }

    // calculate homography matrix
    double** homogEst = homographyEstimation(before, after);
    int numSupporters = 0;

    // test homography matrix reliability for all points
    for(int j=0; j<numFeat; j++) {
      TranslationVector vec = translationVectors.at(j);
      double* origPoint = dvector(0,3);
      origPoint[0] = vec.x1;
      origPoint[1] = vec.y1;
      origPoint[2] = 1;
      double* matchedPoint = dvector(0,3);
      matchedPoint[0] = vec.x2;
      matchedPoint[1] = vec.y2;
      matchedPoint[2] = 1;

      // matrix multiplication
      double* resultVec = matrixMult(homogEst, origPoint);

      // find difference between homography result point and actual result point
      double xDiff = matchedPoint[0] - (resultVec[0]/resultVec[2]);
      double yDiff = matchedPoint[1] - (resultVec[1]/resultVec[2]);
      double dist = sqrt(xDiff*xDiff + yDiff*yDiff);

      if(dist < acceptThresh) {
        numSupporters++;
      }
    }

    // construct homography matrix struct and add to array of possible matricies
    HomographyMat h;
    h.homographyMatrix = homogEst;
    h.supporters = numSupporters;
    homogMatricies.push_back(h);
  }

  std::sort(homogMatricies.begin(), homogMatricies.end(), sortHomogByNumSupporters);
  HomographyMat winner = homogMatricies.at(0);

  //printf("%f %f %f\n%f %f %f\n%f %f %f\n", winner.homographyMatrix[0][0], winner.homographyMatrix[0][1], winner.homographyMatrix[0][2],winner.homographyMatrix[1][0], winner.homographyMatrix[1][1], winner.homographyMatrix[1][2],winner.homographyMatrix[2][0], winner.homographyMatrix[2][1], winner.homographyMatrix[2][2]);

  // set original image opacity to 50%
  for(int x=0; x<width; x++) {
    for(int y=0; y<height; y++) {
      SetPixel(x,y, R2black_pixel);
    }
  }

  // average with other image, skewed by winner H matrix
  double* vec = dvector(0,3);
  for(int x=0; x<otherImage->width; x++) {
    for(int y=0; y<otherImage->height; y++) {
      vec[0] = x;
      vec[1] = y;
      vec[2] = 1;
      double* resultPoint = matrixMult(winner.homographyMatrix, vec);
      double a = resultPoint[0];
      double b = resultPoint[1];
      double c = resultPoint[2];
      //printf("a:%f b:%f c:%f\n", a, b, c);
      double newX = a/c;
      double newY = b/c;
      int newXInt = (int)newX;
      int newYInt = (int)newY;
      //printf("x: %d y: %d || newX: %f newY: %f\n", x, y, newX, newY);

      R2Pixel pixelA = otherImage->Pixel(newXInt,newYInt);
      R2Pixel pixelB = otherImage->Pixel(newXInt+1,newYInt);
      R2Pixel pixelC = otherImage->Pixel(newXInt,newYInt+1);
      R2Pixel pixelD = otherImage->Pixel(newXInt+1,newYInt+1);

      double weightA = (newXInt+1-newX) * (newYInt+1-newY);
      double weightB = (newX-newXInt) * (newYInt+1-newY);
      double weightC = (newXInt+1-newX) * (newY-newYInt);
      double weightD = (newX-newXInt) * (newY-newYInt);

      //printf("A:%f B:%f C:%f D:%f\n", weightA, weightB, weightC, weightD);

      R2Pixel interpolated = weightA*pixelA + weightB*pixelB + weightC*pixelC + weightD*pixelD;

      if(newX < 0 || newY < 0 || newX > width || newY > height) {
        // skip
      }
      else {
        //printf("setting pixel x:%d y:%d\n", x, y);
        SetPixel(x, y, interpolated);
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////
// I/O Functions
////////////////////////////////////////////////////////////////////////

int R2Image::
Read(const char *filename)
{
  // Initialize everything
  if (pixels) { delete [] pixels; pixels = NULL; }
  npixels = width = height = 0;

  // Parse input filename extension
  char *input_extension;
  if (!(input_extension = (char*)strrchr(filename, '.'))) {
	fprintf(stderr, "Input file has no extension (e.g., .jpg).\n");
	return 0;
  }

  // Read file of appropriate type
  if (!strncmp(input_extension, ".bmp", 4)) return ReadBMP(filename);
  else if (!strncmp(input_extension, ".ppm", 4)) return ReadPPM(filename);
  else if (!strncmp(input_extension, ".jpg", 4)) return ReadJPEG(filename);
  else if (!strncmp(input_extension, ".jpeg", 5)) return ReadJPEG(filename);

  // Should never get here
  fprintf(stderr, "Unrecognized image file extension");
  return 0;
}



int R2Image::
Write(const char *filename) const
{
  // Parse input filename extension
  char *input_extension;
  if (!(input_extension = (char*)strrchr(filename, '.'))) {
	fprintf(stderr, "Input file has no extension (e.g., .jpg).\n");
	return 0;
  }

  // Write file of appropriate type
  if (!strncmp(input_extension, ".bmp", 4)) return WriteBMP(filename);
  else if (!strncmp(input_extension, ".ppm", 4)) return WritePPM(filename, 1);
  else if (!strncmp(input_extension, ".jpg", 5)) return WriteJPEG(filename);
  else if (!strncmp(input_extension, ".jpeg", 5)) return WriteJPEG(filename);

  // Should never get here
  fprintf(stderr, "Unrecognized image file extension");
  return 0;
}



////////////////////////////////////////////////////////////////////////
// BMP I/O
////////////////////////////////////////////////////////////////////////

#if (RN_OS == RN_LINUX) && !WIN32

typedef struct tagBITMAPFILEHEADER {
  unsigned short int bfType;
  unsigned int bfSize;
  unsigned short int bfReserved1;
  unsigned short int bfReserved2;
  unsigned int bfOffBits;
} BITMAPFILEHEADER;

typedef struct tagBITMAPINFOHEADER {
  unsigned int biSize;
  int biWidth;
  int biHeight;
  unsigned short int biPlanes;
  unsigned short int biBitCount;
  unsigned int biCompression;
  unsigned int biSizeImage;
  int biXPelsPerMeter;
  int biYPelsPerMeter;
  unsigned int biClrUsed;
  unsigned int biClrImportant;
} BITMAPINFOHEADER;

typedef struct tagRGBTRIPLE {
  unsigned char rgbtBlue;
  unsigned char rgbtGreen;
  unsigned char rgbtRed;
} RGBTRIPLE;

typedef struct tagRGBQUAD {
  unsigned char rgbBlue;
  unsigned char rgbGreen;
  unsigned char rgbRed;
  unsigned char rgbReserved;
} RGBQUAD;

#endif

#define BI_RGB        0L
#define BI_RLE8       1L
#define BI_RLE4       2L
#define BI_BITFIELDS  3L

#define BMP_BF_TYPE 0x4D42 /* word BM */
#define BMP_BF_OFF_BITS 54 /* 14 for file header + 40 for info header (not sizeof(), but packed size) */
#define BMP_BI_SIZE 40 /* packed size of info header */


static unsigned short int WordReadLE(FILE *fp)
{
  // Read a unsigned short int from a file in little endian format
  unsigned short int lsb, msb;
  lsb = getc(fp);
  msb = getc(fp);
  return (msb << 8) | lsb;
}



static void WordWriteLE(unsigned short int x, FILE *fp)
{
  // Write a unsigned short int to a file in little endian format
  unsigned char lsb = (unsigned char) (x & 0x00FF); putc(lsb, fp);
  unsigned char msb = (unsigned char) (x >> 8); putc(msb, fp);
}



static unsigned int DWordReadLE(FILE *fp)
{
  // Read a unsigned int word from a file in little endian format
  unsigned int b1 = getc(fp);
  unsigned int b2 = getc(fp);
  unsigned int b3 = getc(fp);
  unsigned int b4 = getc(fp);
  return (b4 << 24) | (b3 << 16) | (b2 << 8) | b1;
}



static void DWordWriteLE(unsigned int x, FILE *fp)
{
  // Write a unsigned int to a file in little endian format
  unsigned char b1 = (x & 0x000000FF); putc(b1, fp);
  unsigned char b2 = ((x >> 8) & 0x000000FF); putc(b2, fp);
  unsigned char b3 = ((x >> 16) & 0x000000FF); putc(b3, fp);
  unsigned char b4 = ((x >> 24) & 0x000000FF); putc(b4, fp);
}



static int LongReadLE(FILE *fp)
{
  // Read a int word from a file in little endian format
  int b1 = getc(fp);
  int b2 = getc(fp);
  int b3 = getc(fp);
  int b4 = getc(fp);
  return (b4 << 24) | (b3 << 16) | (b2 << 8) | b1;
}



static void LongWriteLE(int x, FILE *fp)
{
  // Write a int to a file in little endian format
  char b1 = (x & 0x000000FF); putc(b1, fp);
  char b2 = ((x >> 8) & 0x000000FF); putc(b2, fp);
  char b3 = ((x >> 16) & 0x000000FF); putc(b3, fp);
  char b4 = ((x >> 24) & 0x000000FF); putc(b4, fp);
}



int R2Image::
ReadBMP(const char *filename)
{
  // Open file
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
	fprintf(stderr, "Unable to open image file: %s", filename);
	return 0;
  }

  /* Read file header */
  BITMAPFILEHEADER bmfh;
  bmfh.bfType = WordReadLE(fp);
  bmfh.bfSize = DWordReadLE(fp);
  bmfh.bfReserved1 = WordReadLE(fp);
  bmfh.bfReserved2 = WordReadLE(fp);
  bmfh.bfOffBits = DWordReadLE(fp);

  /* Check file header */
  assert(bmfh.bfType == BMP_BF_TYPE);
  /* ignore bmfh.bfSize */
  /* ignore bmfh.bfReserved1 */
  /* ignore bmfh.bfReserved2 */
  assert(bmfh.bfOffBits == BMP_BF_OFF_BITS);

  /* Read info header */
  BITMAPINFOHEADER bmih;
  bmih.biSize = DWordReadLE(fp);
  bmih.biWidth = LongReadLE(fp);
  bmih.biHeight = LongReadLE(fp);
  bmih.biPlanes = WordReadLE(fp);
  bmih.biBitCount = WordReadLE(fp);
  bmih.biCompression = DWordReadLE(fp);
  bmih.biSizeImage = DWordReadLE(fp);
  bmih.biXPelsPerMeter = LongReadLE(fp);
  bmih.biYPelsPerMeter = LongReadLE(fp);
  bmih.biClrUsed = DWordReadLE(fp);
  bmih.biClrImportant = DWordReadLE(fp);

  // Check info header
  assert(bmih.biSize == BMP_BI_SIZE);
  assert(bmih.biWidth > 0);
  assert(bmih.biHeight > 0);
  assert(bmih.biPlanes == 1);
  assert(bmih.biBitCount == 24);  /* RGB */
  assert(bmih.biCompression == BI_RGB);   /* RGB */
  int lineLength = bmih.biWidth * 3;  /* RGB */
  if ((lineLength % 4) != 0) lineLength = (lineLength / 4 + 1) * 4;
  assert(bmih.biSizeImage == (unsigned int) lineLength * (unsigned int) bmih.biHeight);

  // Assign width, height, and number of pixels
  width = bmih.biWidth;
  height = bmih.biHeight;
  npixels = width * height;

  // Allocate unsigned char buffer for reading pixels
  int rowsize = 3 * width;
  if ((rowsize % 4) != 0) rowsize = (rowsize / 4 + 1) * 4;
  int nbytes = bmih.biSizeImage;
  unsigned char *buffer = new unsigned char [nbytes];
  if (!buffer) {
	fprintf(stderr, "Unable to allocate temporary memory for BMP file");
	fclose(fp);
	return 0;
  }

  // Read buffer
  fseek(fp, (long) bmfh.bfOffBits, SEEK_SET);
  if (fread(buffer, 1, bmih.biSizeImage, fp) != bmih.biSizeImage) {
	fprintf(stderr, "Error while reading BMP file %s", filename);
	return 0;
  }

  // Close file
  fclose(fp);

  // Allocate pixels for image
  pixels = new R2Pixel [ width * height ];
  if (!pixels) {
	fprintf(stderr, "Unable to allocate memory for BMP file");
	fclose(fp);
	return 0;
  }

  // Assign pixels
  for (int j = 0; j < height; j++) {
	unsigned char *p = &buffer[j * rowsize];
	for (int i = 0; i < width; i++) {
	  double b = (double) *(p++) / 255;
	  double g = (double) *(p++) / 255;
	  double r = (double) *(p++) / 255;
	  R2Pixel pixel(r, g, b, 1);
	  SetPixel(i, j, pixel);
	}
  }

  // Free unsigned char buffer for reading pixels
  delete [] buffer;

  // Return success
  return 1;
}



int R2Image::
WriteBMP(const char *filename) const
{
  // Open file
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
	fprintf(stderr, "Unable to open image file: %s", filename);
	return 0;
  }

  // Compute number of bytes in row
  int rowsize = 3 * width;
  if ((rowsize % 4) != 0) rowsize = (rowsize / 4 + 1) * 4;

  // Write file header
  BITMAPFILEHEADER bmfh;
  bmfh.bfType = BMP_BF_TYPE;
  bmfh.bfSize = BMP_BF_OFF_BITS + rowsize * height;
  bmfh.bfReserved1 = 0;
  bmfh.bfReserved2 = 0;
  bmfh.bfOffBits = BMP_BF_OFF_BITS;
  WordWriteLE(bmfh.bfType, fp);
  DWordWriteLE(bmfh.bfSize, fp);
  WordWriteLE(bmfh.bfReserved1, fp);
  WordWriteLE(bmfh.bfReserved2, fp);
  DWordWriteLE(bmfh.bfOffBits, fp);

  // Write info header
  BITMAPINFOHEADER bmih;
  bmih.biSize = BMP_BI_SIZE;
  bmih.biWidth = width;
  bmih.biHeight = height;
  bmih.biPlanes = 1;
  bmih.biBitCount = 24;       /* RGB */
  bmih.biCompression = BI_RGB;    /* RGB */
  bmih.biSizeImage = rowsize * (unsigned int) bmih.biHeight;  /* RGB */
  bmih.biXPelsPerMeter = 2925;
  bmih.biYPelsPerMeter = 2925;
  bmih.biClrUsed = 0;
  bmih.biClrImportant = 0;
  DWordWriteLE(bmih.biSize, fp);
  LongWriteLE(bmih.biWidth, fp);
  LongWriteLE(bmih.biHeight, fp);
  WordWriteLE(bmih.biPlanes, fp);
  WordWriteLE(bmih.biBitCount, fp);
  DWordWriteLE(bmih.biCompression, fp);
  DWordWriteLE(bmih.biSizeImage, fp);
  LongWriteLE(bmih.biXPelsPerMeter, fp);
  LongWriteLE(bmih.biYPelsPerMeter, fp);
  DWordWriteLE(bmih.biClrUsed, fp);
  DWordWriteLE(bmih.biClrImportant, fp);

  // Write image, swapping blue and red in each pixel
  int pad = rowsize - width * 3;
  for (int j = 0; j < height; j++) {
	for (int i = 0; i < width; i++) {
	  const R2Pixel& pixel = (*this)[i][j];
	  double r = 255.0 * pixel.Red();
	  double g = 255.0 * pixel.Green();
	  double b = 255.0 * pixel.Blue();
	  if (r >= 255) r = 255;
	  if (g >= 255) g = 255;
	  if (b >= 255) b = 255;
	  fputc((unsigned char) b, fp);
	  fputc((unsigned char) g, fp);
	  fputc((unsigned char) r, fp);
	}

	// Pad row
	for (int i = 0; i < pad; i++) fputc(0, fp);
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// PPM I/O
////////////////////////////////////////////////////////////////////////

int R2Image::
ReadPPM(const char *filename)
{
  // Open file
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
	fprintf(stderr, "Unable to open image file: %s", filename);
	return 0;
  }

  // Read PPM file magic identifier
  char buffer[128];
  if (!fgets(buffer, 128, fp)) {
	fprintf(stderr, "Unable to read magic id in PPM file");
	fclose(fp);
	return 0;
  }

  // skip comments
  int c = getc(fp);
  while (c == '#') {
	while (c != '\n') c = getc(fp);
	c = getc(fp);
  }
  ungetc(c, fp);

  // Read width and height
  if (fscanf(fp, "%d%d", &width, &height) != 2) {
	fprintf(stderr, "Unable to read width and height in PPM file");
	fclose(fp);
	return 0;
  }

  // Read max value
  double max_value;
  if (fscanf(fp, "%lf", &max_value) != 1) {
	fprintf(stderr, "Unable to read max_value in PPM file");
	fclose(fp);
	return 0;
  }

  // Allocate image pixels
  pixels = new R2Pixel [ width * height ];
  if (!pixels) {
	fprintf(stderr, "Unable to allocate memory for PPM file");
	fclose(fp);
	return 0;
  }

  // Check if raw or ascii file
  if (!strcmp(buffer, "P6\n")) {
	// Read up to one character of whitespace (\n) after max_value
	int c = getc(fp);
	if (!isspace(c)) putc(c, fp);

	// Read raw image data
	// First ppm pixel is top-left, so read in opposite scan-line order
	for (int j = height-1; j >= 0; j--) {
	  for (int i = 0; i < width; i++) {
		double r = (double) getc(fp) / max_value;
		double g = (double) getc(fp) / max_value;
		double b = (double) getc(fp) / max_value;
		R2Pixel pixel(r, g, b, 1);
		SetPixel(i, j, pixel);
	  }
	}
  }
  else {
	// Read asci image data
	// First ppm pixel is top-left, so read in opposite scan-line order
	for (int j = height-1; j >= 0; j--) {
	  for (int i = 0; i < width; i++) {
	// Read pixel values
	int red, green, blue;
	if (fscanf(fp, "%d%d%d", &red, &green, &blue) != 3) {
	  fprintf(stderr, "Unable to read data at (%d,%d) in PPM file", i, j);
	  fclose(fp);
	  return 0;
	}

	// Assign pixel values
	double r = (double) red / max_value;
	double g = (double) green / max_value;
	double b = (double) blue / max_value;
		R2Pixel pixel(r, g, b, 1);
		SetPixel(i, j, pixel);
	  }
	}
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



int R2Image::
WritePPM(const char *filename, int ascii) const
{
  // Check type
  if (ascii) {
	// Open file
	FILE *fp = fopen(filename, "w");
	if (!fp) {
	  fprintf(stderr, "Unable to open image file: %s", filename);
	  return 0;
	}

	// Print PPM image file
	// First ppm pixel is top-left, so write in opposite scan-line order
	fprintf(fp, "P3\n");
	fprintf(fp, "%d %d\n", width, height);
	fprintf(fp, "255\n");
	for (int j = height-1; j >= 0 ; j--) {
	  for (int i = 0; i < width; i++) {
		const R2Pixel& p = (*this)[i][j];
		int r = (int) (255 * p.Red());
		int g = (int) (255 * p.Green());
		int b = (int) (255 * p.Blue());
		fprintf(fp, "%-3d %-3d %-3d  ", r, g, b);
		if (((i+1) % 4) == 0) fprintf(fp, "\n");
	  }
	  if ((width % 4) != 0) fprintf(fp, "\n");
	}
	fprintf(fp, "\n");

	// Close file
	fclose(fp);
  }
  else {
	// Open file
	FILE *fp = fopen(filename, "wb");
	if (!fp) {
	  fprintf(stderr, "Unable to open image file: %s", filename);
	  return 0;
	}

	// Print PPM image file
	// First ppm pixel is top-left, so write in opposite scan-line order
	fprintf(fp, "P6\n");
	fprintf(fp, "%d %d\n", width, height);
	fprintf(fp, "255\n");
	for (int j = height-1; j >= 0 ; j--) {
	  for (int i = 0; i < width; i++) {
		const R2Pixel& p = (*this)[i][j];
		int r = (int) (255 * p.Red());
		int g = (int) (255 * p.Green());
		int b = (int) (255 * p.Blue());
		fprintf(fp, "%c%c%c", r, g, b);
	  }
	}

	// Close file
	fclose(fp);
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// JPEG I/O
////////////////////////////////////////////////////////////////////////


// #define USE_JPEG
#ifdef USE_JPEG
  extern "C" {
#   define XMD_H // Otherwise, a conflict with INT32
#   undef FAR // Otherwise, a conflict with windows.h
#   include "jpeg/jpeglib.h"
  };
#endif



int R2Image::
ReadJPEG(const char *filename)
{
#ifdef USE_JPEG
  // Open file
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
	fprintf(stderr, "Unable to open image file: %s", filename);
	return 0;
  }

  // Initialize decompression info
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, fp);
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  // Remember image attributes
  width = cinfo.output_width;
  height = cinfo.output_height;
  npixels = width * height;
  int ncomponents = cinfo.output_components;

  // Allocate pixels for image
  pixels = new R2Pixel [ npixels ];
  if (!pixels) {
	fprintf(stderr, "Unable to allocate memory for BMP file");
	fclose(fp);
	return 0;
  }

  // Allocate unsigned char buffer for reading image
  int rowsize = ncomponents * width;
  if ((rowsize % 4) != 0) rowsize = (rowsize / 4 + 1) * 4;
  int nbytes = rowsize * height;
  unsigned char *buffer = new unsigned char [nbytes];
  if (!buffer) {
	fprintf(stderr, "Unable to allocate temporary memory for JPEG file");
	fclose(fp);
	return 0;
  }

  // Read scan lines
  // First jpeg pixel is top-left, so read pixels in opposite scan-line order
  while (cinfo.output_scanline < cinfo.output_height) {
	int scanline = cinfo.output_height - cinfo.output_scanline - 1;
	unsigned char *row_pointer = &buffer[scanline * rowsize];
	jpeg_read_scanlines(&cinfo, &row_pointer, 1);
  }

  // Free everything
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  // Close file
  fclose(fp);

  // Assign pixels
  for (int j = 0; j < height; j++) {
	unsigned char *p = &buffer[j * rowsize];
	for (int i = 0; i < width; i++) {
	  double r, g, b, a;
	  if (ncomponents == 1) {
		r = g = b = (double) *(p++) / 255;
		a = 1;
	  }
	  else if (ncomponents == 1) {
		r = g = b = (double) *(p++) / 255;
		a = 1;
		p++;
	  }
	  else if (ncomponents == 3) {
		r = (double) *(p++) / 255;
		g = (double) *(p++) / 255;
		b = (double) *(p++) / 255;
		a = 1;
	  }
	  else if (ncomponents == 4) {
		r = (double) *(p++) / 255;
		g = (double) *(p++) / 255;
		b = (double) *(p++) / 255;
		a = (double) *(p++) / 255;
	  }
	  else {
		fprintf(stderr, "Unrecognized number of components in jpeg image: %d\n", ncomponents);
		return 0;
	  }
	  R2Pixel pixel(r, g, b, a);
	  SetPixel(i, j, pixel);
	}
  }

  // Free unsigned char buffer for reading pixels
  delete [] buffer;

  // Return success
  return 1;
#else
  fprintf(stderr, "JPEG not supported");
  return 0;
#endif
}




int R2Image::
WriteJPEG(const char *filename) const
{
#ifdef USE_JPEG
  // Open file
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
	fprintf(stderr, "Unable to open image file: %s", filename);
	return 0;
  }

  // Initialize compression info
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, fp);
  cinfo.image_width = width; 	/* image width and height, in pixels */
  cinfo.image_height = height;
  cinfo.input_components = 3;		/* # of color components per pixel */
  cinfo.in_color_space = JCS_RGB; 	/* colorspace of input image */
  cinfo.dct_method = JDCT_ISLOW;
  jpeg_set_defaults(&cinfo);
  cinfo.optimize_coding = TRUE;
  jpeg_set_quality(&cinfo, 95, TRUE);
  jpeg_start_compress(&cinfo, TRUE);

  // Allocate unsigned char buffer for reading image
  int rowsize = 3 * width;
  if ((rowsize % 4) != 0) rowsize = (rowsize / 4 + 1) * 4;
  int nbytes = rowsize * height;
  unsigned char *buffer = new unsigned char [nbytes];
  if (!buffer) {
	fprintf(stderr, "Unable to allocate temporary memory for JPEG file");
	fclose(fp);
	return 0;
  }

  // Fill buffer with pixels
  for (int j = 0; j < height; j++) {
	unsigned char *p = &buffer[j * rowsize];
	for (int i = 0; i < width; i++) {
	  const R2Pixel& pixel = (*this)[i][j];
	  int r = (int) (255 * pixel.Red());
	  int g = (int) (255 * pixel.Green());
	  int b = (int) (255 * pixel.Blue());
	  if (r > 255) r = 255;
	  if (g > 255) g = 255;
	  if (b > 255) b = 255;
	  *(p++) = r;
	  *(p++) = g;
	  *(p++) = b;
	}
  }



  // Output scan lines
  // First jpeg pixel is top-left, so write in opposite scan-line order
  while (cinfo.next_scanline < cinfo.image_height) {
	int scanline = cinfo.image_height - cinfo.next_scanline - 1;
	unsigned char *row_pointer = &buffer[scanline * rowsize];
	jpeg_write_scanlines(&cinfo, &row_pointer, 1);
  }

  // Free everything
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  // Close file
  fclose(fp);

  // Free unsigned char buffer for reading pixels
  delete [] buffer;

  // Return number of bytes written
  return 1;
#else
  fprintf(stderr, "JPEG not supported");
  return 0;
#endif
}
