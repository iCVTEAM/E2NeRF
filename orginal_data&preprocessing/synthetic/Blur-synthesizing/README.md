# Generation of blur image

## Introduction

According to the inverse ISP and ISP process, generate the original image sequence into a image with motion blur.

## Usage

### Parameters

* input_dir: input frames dir
* output_dir: output frame name
* scale_factor: convert scale_factor frame to 1 frame with blur, note that scale_factor must be divide by total number of frames of input
* input_exposure: base exposure time of input frames in microsecond
* input_iso: assumed ISO for input data
* output_iso: expected ISO for output

### Sample

```
python main.py 
    --input_dir ../blender-synthetic-images/chair/r_0/
    --output_name ../blender-synthetic-images/chair/r_0.png
    --scale_factor 18
    --input_exposure 10
    --input_iso 50
    --output_iso 50
```
