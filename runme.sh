#!/bin/bash
#
# Create all output results
#

# Useful shell settings:

# abort the script if a command fails
set -e

# abort the script if an unitialized shell variable is used
set -u

# make sure the code is up to date

pushd src
make
popd

# generate the result pictures


# ~/Downloads/ffmpeg -i testSequence/Exports/Bourne1a.mp4 testSequence/Exports/Bourne1a.%07d.jpg;
# src/imgpro testSequence/Exports/Bourne1a.0000001.jpg output/Bourne1a.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne1a.%07d_ouput.jpg output/Bourne1a_finalResult.m4v

~/Downloads/ffmpeg -i testSequence/Exports/Bourne1b.mp4 testSequence/Exports/Bourne1b.%07d.jpg;
src/imgpro testSequence/Exports/Bourne1b.0000001.jpg output/Bourne1b.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne1b.%07d_ouput.jpg output/Bourne1b_finalResult.m4v

~/Downloads/ffmpeg -i testSequence/Exports/Bourne1c.mp4 testSequence/Exports/Bourne1c.%07d.jpg;
src/imgpro testSequence/Exports/Bourne1c.0000001.jpg output/Bourne1c.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne1c.%07d_ouput.jpg output/Bourne1c_finalResult.m4v

~/Downloads/ffmpeg -i testSequence/Exports/Bourne2a.mp4 testSequence/Exports/Bourne2a.%07d.jpg;
src/imgpro testSequence/Exports/Bourne2a.0000001.jpg output/Bourne2a.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne2a.%07d_ouput.jpg output/Bourne2a_finalResult.m4v

~/Downloads/ffmpeg -i testSequence/Exports/Bourne2b.mp4 testSequence/Exports/Bourne2b.%07d.jpg;
src/imgpro testSequence/Exports/Bourne2b.0000001.jpg output/Bourne2b.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne2b.%07d_ouput.jpg output/Bourne2b_finalResult.m4v

~/Downloads/ffmpeg -i testSequence/Exports/Bourne3.mp4 testSequence/Exports/Bourne3.%07d.jpg;
src/imgpro testSequence/Exports/Bourne3.0000001.jpg output/Bourne3.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne3.%07d_ouput.jpg output/Bourne3_finalResult.m4v

~/Downloads/ffmpeg -i testSequence/Exports/Bourne4a.mp4 testSequence/Exports/Bourne4a.%07d.jpg;
src/imgpro testSequence/Exports/Bourne4a.0000001.jpg output/Bourne4a.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne4a.%07d_ouput.jpg output/Bourne4a_finalResult.m4v

~/Downloads/ffmpeg -i testSequence/Exports/Bourne4b.mp4 testSequence/Exports/Bourne4b.%07d.jpg;
src/imgpro testSequence/Exports/Bourne4b.0000001.jpg output/Bourne4b.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne4b.%07d_ouput.jpg output/Bourne4b_finalResult.m4v

~/Downloads/ffmpeg -i testSequence/Exports/Bourne4c.mp4 testSequence/Exports/Bourne4c.%07d.jpg;
src/imgpro testSequence/Exports/Bourne4c.0000001.jpg output/Bourne4c.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne4c.%07d_ouput.jpg output/Bourne4c_finalResult.m4v

~/Downloads/ffmpeg -i testSequence/Exports/Bourne5.mp4 testSequence/Exports/Bourne5.%07d.jpg;
src/imgpro testSequence/Exports/Bourne5.0000001.jpg output/Bourne5.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne5.%07d_ouput.jpg output/Bourne5_finalResult.m4v

~/Downloads/ffmpeg -i testSequence/Exports/Bourne6.mp4 testSequence/Exports/Bourne6.%07d.jpg;
src/imgpro testSequence/Exports/Bourne6.0000001.jpg output/Bourne6.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne6.%07d_ouput.jpg output/Bourne6_finalResult.m4v

~/Downloads/ffmpeg -i testSequence/Exports/Bourne7.mp4 testSequence/Exports/Bourne7.%07d.jpg;
src/imgpro testSequence/Exports/Bourne7.0000001.jpg output/Bourne7.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne7.%07d_ouput.jpg output/Bourne7_finalResult.m4v

~/Downloads/ffmpeg -i testSequence/Exports/Bourne8.mp4 testSequence/Exports/Bourne8.%07d.jpg;
src/imgpro testSequence/Exports/Bourne8.0000001.jpg output/Bourne8.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne8.%07d_ouput.jpg output/Bourne8_finalResult.m4v

~/Downloads/ffmpeg -i testSequence/Exports/Bourne9.mp4 testSequence/Exports/Bourne9.%07d.jpg;
src/imgpro testSequence/Exports/Bourne9.0000001.jpg output/Bourne9.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne9.%07d_ouput.jpg output/Bourne9_finalResult.m4v

~/Downloads/ffmpeg -i testSequence/Exports/Bourne10.mp4 testSequence/Exports/Bourne10.%07d.jpg;
src/imgpro testSequence/Exports/Bourne10.0000001.jpg output/Bourne10.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne10.%07d_ouput.jpg output/Bourne10_finalResult.m4v

~/Downloads/ffmpeg -i testSequence/Exports/Bourne11.mp4 testSequence/Exports/Bourne11.%07d.jpg;
src/imgpro testSequence/Exports/Bourne11.0000001.jpg output/Bourne11.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne11.%07d_ouput.jpg output/Bourne11_finalResult.m4v

~/Downloads/ffmpeg -i testSequence/Exports/Bourne12.mp4 testSequence/Exports/Bourne12.%07d.jpg;
src/imgpro testSequence/Exports/Bourne12.0000001.jpg output/Bourne12.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne12.%07d_ouput.jpg output/Bourne12_finalResult.m4v

~/Downloads/ffmpeg -i testSequence/Exports/Bourne13.mp4 testSequence/Exports/Bourne13.%07d.jpg;
src/imgpro testSequence/Exports/Bourne13.0000001.jpg output/Bourne13.0000001_output.jpg -stabilize 30.0;
~/Downloads/ffmpeg -i output/Bourne13.%07d_ouput.jpg output/Bourne13_finalResult.m4v
