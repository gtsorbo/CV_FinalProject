
// Computer Vision for Digital Post-Production
// Lecturer: Gergely Vass - vassg@vassg.hu
//
// Skeleton Code for programming assigments
//
// Code originally from Thomas Funkhouser
// main.c
// original by Wagner Correa, 1999
// modified by Robert Osada, 2000
// modified by Renato Werneck, 2003
// modified by Jason Lawrence, 2004
// modified by Jason Lawrence, 2005
// modified by Forrester Cole, 2006
// modified by Tom Funkhouser, 2007
// modified by Chris DeCoro, 2007
//



// Include files

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "R2/R2.h"
#include "R2Pixel.h"
#include "R2Image.h"
#include <string>



// Program arguments

static char options[] =
"  -help\n"
"  -svdTest\n"
"  -sobelX\n"
"  -sobelY\n"
"  -log\n"
"  -harris <real:sigma>\n"
"  -saturation <real:factor>\n"
"  -brightness <real:factor>\n"
"  -blur <real:sigma>\n"
"  -sharpen \n"
"  -matchTranslation <file:other_image>\n"
"  -matchHomography <file:other_image>\n"
"  -stabilize\n";


static void
ShowUsage(void)
{
  // Print usage message and exit
  fprintf(stderr, "Usage: imgpro input_image output_image [  -option [arg ...] ...]\n");
  fprintf(stderr, options);
  exit(EXIT_FAILURE);
}



static void
CheckOption(char *option, int argc, int minargc)
{
  // Check if there are enough remaining arguments for option
  if (argc < minargc)  {
    fprintf(stderr, "Too few arguments for %s\n", option);
    ShowUsage();
    exit(-1);
  }
}



static int
ReadCorrespondences(char *filename, R2Segment *&source_segments, R2Segment *&target_segments, int& nsegments)
{
  // Open file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open correspondences file %s\n", filename);
    exit(-1);
  }

  // Read number of segments
  if (fscanf(fp, "%d", &nsegments) != 1) {
    fprintf(stderr, "Unable to read correspondences file %s\n", filename);
    exit(-1);
  }

  // Allocate arrays for segments
  source_segments = new R2Segment [ nsegments ];
  target_segments = new R2Segment [ nsegments ];
  if (!source_segments || !target_segments) {
    fprintf(stderr, "Unable to allocate correspondence segments for %s\n", filename);
    exit(-1);
  }

  // Read segments
  for (int i = 0; i <  nsegments; i++) {

    // Read source segment
    double sx1, sy1, sx2, sy2;
    if (fscanf(fp, "%lf%lf%lf%lf", &sx1, &sy1, &sx2, &sy2) != 4) {
      fprintf(stderr, "Error reading correspondence %d out of %d\n", i, nsegments);
      exit(-1);
    }

    // Read target segment
    double tx1, ty1, tx2, ty2;
    if (fscanf(fp, "%lf%lf%lf%lf", &tx1, &ty1, &tx2, &ty2) != 4) {
      fprintf(stderr, "Error reading correspondence %d out of %d\n", i, nsegments);
      exit(-1);
    }

    // Add segments to list
    source_segments[i] = R2Segment(sx1, sy1, sx2, sy2);
    target_segments[i] = R2Segment(tx1, ty1, tx2, ty2);
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



int
main(int argc, char **argv)
{
  // Look for help
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "-help")) {
      ShowUsage();
    }
	if (!strcmp(argv[i], "-svdTest")) {
      R2Image *image = new R2Image();
	  image->svdTest();
	  return 0;
    }
  }

  // Read input and output image filenames
  if (argc < 3)  ShowUsage();
  argv++, argc--; // First argument is program name
  char *input_image_name = *argv; argv++, argc--;
  char *output_image_name = *argv; argv++, argc--;

  // Allocate image
  R2Image *image = new R2Image();
  if (!image) {
    fprintf(stderr, "Unable to allocate image\n");
    exit(-1);
  }

  // Read input image
  if (!image->Read(input_image_name)) {
    fprintf(stderr, "Unable to read image from %s\n", input_image_name);
    exit(-1);
  }

  // Initialize sampling method
  int sampling_method = R2_IMAGE_POINT_SAMPLING;

  // Parse arguments and perform operations
  while (argc > 0) {
    if (!strcmp(*argv, "-brightness")) {
      CheckOption(*argv, argc, 2);
      double factor = atof(argv[1]);
      argv += 2, argc -=2;
      image->Brighten(factor);
    }
	   else if (!strcmp(*argv, "-sobelX")) {
      argv++, argc--;
      image->SobelX();
    }
	   else if (!strcmp(*argv, "-sobelY")) {
      argv++, argc--;
      image->SobelY();
    }
	   else if (!strcmp(*argv, "-log")) {
      argv++, argc--;
      image->LoG();
    }
    else if (!strcmp(*argv, "-saturation")) {
      CheckOption(*argv, argc, 2);
      double factor = atof(argv[1]);
      argv += 2, argc -= 2;
      image->ChangeSaturation(factor);
    }
	   else if (!strcmp(*argv, "-harris")) {
      CheckOption(*argv, argc, 2);
      double sigma = atof(argv[1]);
      argv += 2, argc -= 2;
      image->Harris(sigma);
    }
    else if (!strcmp(*argv, "-blur")) {
      CheckOption(*argv, argc, 2);
      double sigma = atof(argv[1]);
      argv += 2, argc -= 2;
      image->Blur(sigma);
    }
    else if (!strcmp(*argv, "-sharpen")) {
      argv++, argc--;
      image->Sharpen();
    }
    else if (!strcmp(*argv, "-matchTranslation")) {
      CheckOption(*argv, argc, 2);
      R2Image *other_image = new R2Image(argv[1]);
      argv += 2, argc -= 2;
      //image->blendOtherImageTranslated(other_image);
      delete other_image;
    }
    else if (!strcmp(*argv, "-matchHomography")) {
      CheckOption(*argv, argc, 2);
      R2Image *other_image = new R2Image(argv[1]);
      argv += 2, argc -= 2;
      image->blendOtherImageHomography(other_image);
      delete other_image;
    }
    else if(!strcmp(*argv, "-stabilize")) {
      CheckOption(*argv, argc, 2);
      double strength = atof(argv[1]);

      argv += 2, argc -= 2;

      // parse file
      std::string file = input_image_name;
      std::string out = output_image_name;
      std::string delim = ".";
      std::string token = file.substr(0, file.find(delim)); // beginnign of file
      std::string token_output = out.substr(0, out.find(delim));
      std::string temp = file.substr(file.find(delim)+1, file.length());
      std::string count = temp.substr(0, temp.find(delim)); // start count
      //printf("%s  %s  %f\n", token.c_str(), count.c_str(), strength);
      int count_length = count.length();
      int current_count = std::stoi(count);
      int start_count = current_count;
      R2Image *other_image = new R2Image();
      std::string image_count_1 = std::to_string(current_count);
      std::string image_count_2 = std::to_string(current_count+1);
      // fix file counter string
      while(image_count_1.length() < count_length) {
        image_count_1 = "0" + image_count_1;
      }
      while(image_count_2.length() < count_length) {
        image_count_2 = "0" + image_count_2;
      }
      std::string image_name_1 = token + "." + image_count_1 + ".jpg";
      std::string image_name_2 = token + "." + image_count_2 + ".jpg";
      std::string output_name = token_output + "." + image_count_1 + "_output.jpg";
      strcpy(output_image_name, output_name.c_str());

      std::vector<ContextPixel> foundFeatures = image->findBestFeatures();
      std::vector<ContextPixel> matchedFeatures;
      std::vector<TranslationVector> motionVectors;

      // Process translations
      while (image->Read(image_name_1.c_str()) && other_image->Read(image_name_2.c_str())) {
        printf("%s  %s\n", image_name_1.c_str(), image_name_2.c_str());

        current_count++;

        // Filename incrementing
        image_count_1 = std::to_string(current_count);
        image_count_2 = std::to_string(current_count+1);
        while(image_count_1.length() < count_length) {
          image_count_1 = "0" + image_count_1;
        }
        while(image_count_2.length() < count_length) {
          image_count_2 = "0" + image_count_2;
        }
        image_name_1 = token + "." + image_count_1 + ".jpg";
        image_name_2 = token + "." + image_count_2 + ".jpg";
        output_name = token_output + "." + image_count_1 + "_ouput.jpg";
        strcpy(output_image_name, output_name.c_str());

        // find matched features in next image
        matchedFeatures = image->blendOtherImageTranslated(other_image, foundFeatures);
        std::vector<TranslationVector> vectors = image-> vectorRANSAC(foundFeatures, matchedFeatures);
        double avg_x = 0.0;
        double avg_y = 0.0;

        // find average motion between frames
        for(int i=0; i<vectors.size(); i++) {
          TranslationVector vec = vectors.at(i);
          avg_x += vec.x2 - vec.x1;
          avg_y += vec.y2 - vec.y1;
        }
        avg_x = avg_x/vectors.size();
        avg_y = avg_y/vectors.size();

        TranslationVector motionVec;
        motionVec.x = avg_x;
        motionVec.y = avg_y;
        motionVectors.push_back(motionVec);

        // create context pixels to pass along track points
        std::vector<ContextPixel> RANSACDmatchedFeatures;
        for(int i=0; i<vectors.size(); i++) {
          ContextPixel pix;
          TranslationVector vec = vectors.at(i);
          pix.x = vec.x2;
          pix.y = vec.y2;
          for(int j=0; j<matchedFeatures.size(); j++) {
            ContextPixel matchPix;
            if(matchPix.x == pix.x && matchPix.y == pix.y) {
              pix.pixel = matchPix.pixel;
            }
          }
          RANSACDmatchedFeatures.push_back(pix);
        }

        foundFeatures = RANSACDmatchedFeatures;
        //printf("x1: %d y1: %d x2: %d y2: %d\n", winner.x1, winner.y1, winner.x2, winner.y2);

        //image->Write(output_image_name);
      }

/*
      // Smoothing function
      double x_smoothed[motionVectors.size()];
      double y_smoothed[motionVectors.size()];

      int blur_width = strength;

      for(int i=0; i<motionVectors.size(); i++) {
        // find avg x, y value
        double x_sum = 0.0;
        double y_sum = 0.0;

        for(int j=-1*blur_width; j<blur_width+1; j++) {
          int fixedIndex = i+j;
          if(i+j < 0) {
            fixedIndex = i;
          }
          else if(i+j >= motionVectors.size()) {
            fixedIndex = motionVectors.size()-1;
          }
          TranslationVector vec = motionVectors.at(fixedIndex);
          x_sum += vec.x;
          y_sum += vec.y;
        }

        x_smoothed[i] = x_sum / blur_width*2+1;
        y_smoothed[i] = y_sum / blur_width*2+1;
        //printf("x smoothed: %f, y smoothed:%f\n", x_smoothed[i], y_smoothed[i]);
      }
      */

      // plotting motion curve
      double x_curve[motionVectors.size()+1];
      double y_curve[motionVectors.size()+1];

      x_curve[0] = 0.0;
      y_curve[0] = 0.0;

      for(int i=0; i<motionVectors.size(); i++) {
        TranslationVector vec = motionVectors.at(i);
        x_curve[i+1] = x_curve[i] + vec.x;
        y_curve[i+1] = y_curve[i] + vec.y;
      }

      // find average position in sequence (for still motion)
      double x_avg = 0.0;
      double y_avg = 0.0;

      for(int i=0; i<=motionVectors.size(); i++) {
        x_avg += x_curve[i];
        y_avg += y_curve[i];
      }

      x_avg = x_avg / motionVectors.size()+1;
      y_avg = y_avg / motionVectors.size()+1;
      printf("x_avg %f y_avg %f\n", x_avg, y_avg);


      // apply smoothing and output images
      // second pass over files

      // Process translations
      for (int i=0; i<motionVectors.size(); i++) {
        int current_count = start_count + i;

        // Filename incrementing
        image_count_1 = std::to_string(current_count);
        while(image_count_1.length() < count_length) {
          image_count_1 = "0" + image_count_1;
        }
        image_name_1 = token + "." + image_count_1 + ".jpg";
        output_name = token_output + "." + image_count_1 + "_ouput.jpg";
        strcpy(output_image_name, output_name.c_str());

        printf("before new read\n");
        image->Read(image_name_1.c_str());
        printf("after new read\n");

        // run stabilization per frame
        printf("Adjusting: %s\n", image_name_1.c_str());
        //image->translateImageForStabilization(motionVectors.at(i), x_smoothed[i], y_smoothed[i]);
        image->translateImageForStabilization(x_curve[i], y_curve[i], x_avg, y_avg);
        printf("after stabilization\n");
        printf("about to write\n");
        image->Write(output_image_name);
        printf("after write\n");
      }


    }
    else {
      // Unrecognized program argument
      fprintf(stderr, "image: invalid option: %s\n", *argv);
      ShowUsage();
    }
  }

  // Write output image
  if (!image->Write(output_image_name)) {
    fprintf(stderr, "Unable to read image from %s\n", output_image_name);
    exit(-1);
  }

  // Delete image
  delete image;

  // Return success
  return EXIT_SUCCESS;
}
