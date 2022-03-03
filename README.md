# ND-PSU-3DCartilageSegmentation


This is the repo for 3D Cartilage segmentation. The following discription will walk you through the process of using it.


Part 1: Requirements
The codes best run with tensorflow 1.15, CUDA 10.0 and CUDAnn 7.4. We recommend using the same versions to avoid the compatibility issues. The models are trained on V100 GPUs, so we recommend using that. With smaller batch sizes other GPU may work too, but we have not tested that.

We highly recommend using **conda environment** with python 3.6 to prevent compatibilty issues. You can follow the tutorial in the official website to create the conda environment: https://docs.anaconda.com/anaconda/install/

Install all the requirements using:

**$ pip install -r requirements.txt**


Part 2: Directories
Unless changed in the config.yaml file, the default beaviour of the program expects the data to be in the following order:

img
  - train
      - img
          - E150_001
              - E150_001_0000
              - E150_001_0001
              :
              :
              :
      - mask
          - E150_001
              - E150_001_0000
              - E150_001_0001
              :
              :
              :
  - test
      - E150_001
          - E150_001_0000
          - E150_001_0001
          :
          :
          :

You can change the directories as you prefer but make sure, in the "train" folder you create directiories named img and mask on the same level and put the corresponding training stacks. 

The images in these directories are expected to be in png format. If you have a tif image, you can easily convert them to PNG using the prprocess.py script. Simply put in the source and destination directiries in the config.yaml file and run the script as:

$ python3 preprocess.py

Part 3: Training

We recommended submitting the job to the GPU clusters for training and testing as they could easilytake from 3 to 10 hours to complete. Once the data is correctly set, images are converted to expected png format, you can update the train section of config.yaml. Then update train.sh with the gpu information and notification alerts and submit the job as:

$ qsub train.sh

If for some reason you want to proceed without the job sumbissions, run the train.py script as:

$ python3 train.py

Part 4: Inference

After the models are trained, or you want to infer with the previously trained model, you can change the test section of config.yaml with correct stack name (with its path), specify the model to use (with its path). Then you can update the test.sh with the gpu information and notification alerts and submit the job as:

$ qsub test.sh

If for some reason you want to proceed without the job sumbissions, run the test.py script as:

$ python3 test.py

Part 4: Experiment Tracking

We have options to enable experiment tracking through wandb. You may opt out of it for simplicity. 
