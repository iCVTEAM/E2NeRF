# E<sup>2</sup>NeRF: Event Enhanced Neural Radiance Fields from Blurry Images

## Code
The code is based on [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch).


## Dataset
Download the dataset [here](https://drive.google.com/drive/folders/1XhOEp4UdLL7EnDNyWdxxX8aRvzF53fWo?usp=sharing).
The dataset contains "data" for training and "original data"

### Data
Synthetic Data: There are training images in "train" file and the corresponding event data "events.pt" in each scene file. The ground truth images are in the "test" file. Like in original NeRF, the poses are in the "transform_train.json" file and "transform_test.json" file.

Real-World Data: The structure is like original NeRF's llff data and the event data is in "event.pt". 

### Original Data & Preproccesing
Synthetic Data: There are original images for synthesizing the blurry image and the code. Besides, we supply the original event data generated from v2e. We also provide the code to transform the ".txt" event to "events.pt" for E<sup>2</sup>NeRF training.

Real-World Data: We supply the original ".aedat4" data captured by davis346 and the processing code in the file. We also convert the event data into events.pt for training. 