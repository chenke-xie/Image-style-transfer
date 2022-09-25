
1.CLIP install:


install PyTorch 1.7.1 (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

	$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
	$ pip install ftfy regex tqdm
	$ pip install git+https://github.com/openai/CLIP.git

Replace cudatoolkit=11.0 above with the appropriate CUDA version on your machine or cpuonly when installing on a machine without a GPU.


Alternative:
Install CLIP directly from github. this step may fail.
The solution is to pull the full zip package of the CLIP project from the github, 
save the downloaded CLIP-main.zip file in the local path, and then install the CLIP library directly from the local area.
The specific codes are as follows：
# Go to the path where CLIP-main.zip is located
# Unzip the .zip file, then go to the unzipped folder
unzip CLIP-main.zip
cd CLIP-main
# Run the setup.py file to complete the local installation of clip
python setup.py instal