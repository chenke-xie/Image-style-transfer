# Image Style Transfer
This project replaces the model in the [Image Style Transfer Using Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html), using the RNN and ViT models from [CLIP](https://arxiv.org/abs/2103.00020). A detailed comparison of their performance and the influence of the new model to the original style transfer algorithm are stated in report_Chenke Xie.pdf.

## CLIP install
install [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/)(or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

Replace cudatoolkit=11.0 above with the appropriate CUDA version on your machine or cpuonly when installing on a machine without a GPU.


**Alternative:**
If the above steps fail, download [CLIP](https://github.com/openai/CLIP) directly from Github
pull the full zip package of the CLIP project from the github, save the downloaded CLIP-main.zip file in the local path, and then install the CLIP library directly from the local area.

The steps are as follows:
1. Go to the path where CLIP-main.zip is located
2. Unzip the .zip file, then go to the unzipped folder
```bash
unzip CLIP-main.zip
cd CLIP-main
```
3. Run the setup.py file to complete the local installation of clip
```bash
python setup.py install
```