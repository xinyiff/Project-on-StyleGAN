# Team members and Slides

## Team members
Yuhan Hu      yuhann@bu.edu

Xinyi Feng    fxinyi@bu.edu

## Slides

[523_presentation] (https://docs.google.com/presentation/d/1GukFoZl4_XwSORjsqIby7AcEuLKiJCutCwtyETlwi2A/edit?usp=sharing)

# Generating Images with StyleGAN

In this project, we are going to introduce the basic idea of StyleGAN. Generative Adversarial Networks, or GANs for short, have appeared frequently in the media, which showcasing their ability to generate large high-quality images. In 2018, NVDIA introduce a Style Generative Adersarial Network, or StyleGAN for short, which was one significant step forward for realistic face generation. Next further, StyleGAN was followed by StyleGAN2 in 2019, which improved the quality of StyleGAN by removing certain srtifacts. For the experiment, we will see both how to train StyleGAN2 on any arbitrary set of images; as well as use different pretained weights on Google Colab and local enviroment.

## StyleGAN Architecture 

### Main Method
- Mapping network
The Mapping Network consists of 8 fully connected layers and its output ⱳ is of the same size as the input layer.
The Mapping Network’s goal is to encode the input vector into an intermediate vector whose different elements control different visual features. As a result, the model isn’t capable of mapping parts of the input to features, a phenomenon called features entanglement. So, by using another neural network the model can generate a vector that doesn’t have to follow the training data distribution and can reduce the correlation between features.

- Style Model 
The AdaIN module transfers the encoded information ⱳ, created by the Mapping Network, into the generated image. The module is added to each resolution level of the Synthesis Network and defines the visual expression of the features in that level:
1. Each channel of the convolution layer output is normalized.
2. The intermediate vector ⱳ is transformed using another fully connected layer (marked as A) into a scale and bias for each channel.
3. The scale and bias vectors shift each channel of the convolution output.

- Delete traditional input
Most models use random input to create the initial image of the generator. The StyleGAN found that the image features are controlled by ⱳ and the AdaIN, so the initial input can be omitted and replaced by constant values. So, it’s easier for the network to learn only using ⱳ without relying on the entangled input vector.

- Stochastic variation	
There are many features in people’s faces that are small and can be seen as stochastic, such as freckles and wrinkles, these features make the image more realistic and increase the variety of outputs. The common method to insert these small features into GAN images is adding random noise to the input vector. 
The noise in StyleGAN is added in a similar way to the AdaIN mechanism:  Add a scaled noise to each channel before the AdaIN module and change a bit the visual expression of the features of the resolution level it operates on.

- Style mixing	
The StyleGAN generator uses the intermediate vector in each level of the synthesis network, which might cause the network to learn that levels are correlated. To reduce the correlation, the model randomly selects two input vectors and generates the intermediate vector ⱳ for them. It then trains some of the levels with the first and switches to the other to train the rest of the levels. 
This concept has the ability to combine multiple images in a coherent way. The model generates two images A and B and then combines them by taking low-level features from A and the rest of the features from B.

- Truncation in W
One of the challenges in generative models is dealing with areas that are poorly represented in the training data. The generator isn’t able to learn them and create images that resemble them. To avoid generating poor images, StyleGAN truncates the intermediate vector ⱳ, forcing it to stay close to the “average” intermediate vector.
After training the model, an “average” ⱳ is produced by selecting many random inputs; then generating their intermediate vectors with the mapping network; and calculating the mean of these vectors. 

## Architecture
A traditional generator feeds the latent code through the input layer only, the StyleGAN first maps the input to an intermediate latent space W, which then controls the generator through adaptive instance normalization at each convolution layer. Then the noise is added after each convolution before evaluating the nonlinearity. Here “A” stands for a learned affine transform, and “B” applies learned per-channel scaling factors to the noise input. The output of the last layer is converted to RGB using a separate 1 × 1 convolution. (Picture Source: https://arxiv.org/abs/1812.04948)
![image](https://user-images.githubusercontent.com/86414327/124098732-66836e00-da8f-11eb-978f-eb78ec2c2810.png)

## StyleGAN2 Architecture

Based on the StyleGAN, researchers led by Tero Karras have published a paper where they analyze the capabilities of the original StyleGAN architecture and propose a new improved version -- StyleGAN2. They propose several modifications both in architecture and the training strategy.

<img width="729" alt="Screen Shot 2021-06-28 at 9 43 35 PM" src="https://user-images.githubusercontent.com/70667153/124073815-20211580-da75-11eb-92a9-3b3ee57e4cc2.png">

(Picture Source: https://arxiv.org/pdf/1912.04958.pdf)

<img width="676" alt="Screen Shot 2021-06-29 at 4 48 15 PM" src="https://user-images.githubusercontent.com/70667153/124072882-f5828d00-da73-11eb-9cb5-7b0d8a7f502d.png">

(Picture Source: https://arxiv.org/pdf/1912.04958.pdf)

In particular, researchers redesigned the generator normalization, they revisited the progressive growing as the training stabilization and introduced a new regularization technique to improve conditional generation. The generator architecture was modified such that AdaIn layers were removed i.e adaptive instance normalization was replaced with a “demodulation” operation. After inspecting the effects of progressive growing as a procedure for training with large resolution images, researchers propose an alternative approach where training starts by focusing on low-resolution images and then progressively shifts focus to higher and higher resolutions but without changing the network topology.


## Training Method

- Weight Demodulation

StyleGAN has an issue of Blob like artifact. In this case, resultant images had some unwanted noise which occurred in different locations. And it occurs within synthesis network originating from 64×64 feature maps and finally propagating into output images. To solve this problem, the generator architecture was modified such that AdaIn（adaptive instance normalization) layers were removed. Instead, they introduced Weight Demodulation

- Lazy Regulerization

StyleGANs cost function include computing both main loss function + regularization for every mini-batch. This computation has heavy memory usage and computation cost which could be reduced by only computing regularization term once after 16 mini-batches. This strategy had no drastic changes on model efficiency and thus was being implemented in StyleGAN2.

- Path Length Regularization

It is a type of regularization that allows good conditioning in the mapping from latent codes to images. The idea is to encourage that a fixed-size step in the latent space W results in a non-zero, fixed-magnitude change in the image. 

- Removing Progressive growing

<img width="630" alt="Screen Shot 2021-06-29 at 9 58 05 PM" src="https://user-images.githubusercontent.com/70667153/124074555-3d0a1880-da76-11eb-9e28-e34eca65974d.png">

(Picture Source: https://arxiv.org/pdf/1912.04958.pdf)

As the figure shows, progressive growing leads to “phase” artifacts. In this example the teeth do not follow the pose but stay aligned to the camera, as indicated by the blue line. So the researchers use a hierarchical generator with skip connection (similar to MSG-GAN) instead of progressive growing. In this way, phase artifacts are reduced.

## StyleGAN Implementation with TensorFlow

### Requirements
- Both Linux and Windows are supported. Linux is recommended for performance and compatibility reasons.
- 64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.14.3 or newer.
- We recommend TensorFlow 1.14, which we used for all experiments in the paper, but TensorFlow 1.15 is also supported on Linux. TensorFlow 2.x is not supported.
- On Windows you need to use TensorFlow 1.14, as the standard 1.15 installation does not include necessary C++ headers.
- One or more high-end NVIDIA GPUs, NVIDIA drivers, CUDA 10.0 toolkit and cuDNN 7.5. To reproduce the results reported in the paper, you need an NVIDIA GPU with at least 16 GB of DRAM.
- Docker users: use the provided Dockerfile to build an image with the required library dependencies.

### Generating Real Person Images on Windows 10

#### Preparing Dataset
To generate the real person faces, we used the FFHQ which refers to Flicker-Faces-HQ Datasets and trained this Datasets using official provided code. Datasets are stored as multi-resolution TFRecords. Each dataset consists of multiple *.tfrecords files stored under a common directory, e.g., ~/datasets/ffhq/ffhq-r*.tfrecords. In the following sections, the datasets are referenced using a combination of --dataset and --data-dir arguments, e.g., --dataset=ffhq --data-dir=~/datasets.

#### Training Network
To train the Dataset, run the run_projector.py file to find the matching latent vectors for a set of images. The training will output a pikle file with the resulting networks.

#### Generated Real Person Images
Here are the generated person faces.
![屏幕截图 2021-06-30 141829](https://user-images.githubusercontent.com/70667153/124163161-e29ea580-dad1-11eb-98d8-120539560855.png)
![屏幕截图 2021-06-30 203101](https://user-images.githubusercontent.com/70667153/124163167-e3cfd280-dad1-11eb-8680-3efa8a1863bd.png)


### Generating Anime Characters on Google Colab

#### Preparing Dataset

In this part, we use the pre-trained Anime StyleGAN2 by Aaron Gokslan so that we can directly load the model and generate the anime images. For the implement convience, we would like to coding on the Google Colab this time. 

The detailed coding can be seen in the StyleGAN2_Animation.ipynb. 
[ipython notebook](https://github.com/xinyiff/Project-on-StyleGAN/blob/ee4e3ffde5707fabee05f5efc85a66bc26a44d33/StyleGAN2_Animation.ipynb)

#### Generated Anime Characters

The first generated images shows below.

![屏幕截图 2021-06-30 201944](https://user-images.githubusercontent.com/70667153/124075687-da198100-da77-11eb-94f9-57401a3597f9.png)

We also can show it in a grid of images to see multiple images at one time. Below is the generated images in a 3 * 3 grid.
![屏幕截图 2021-06-30 201827](https://user-images.githubusercontent.com/70667153/124075757-ee5d7e00-da77-11eb-95f1-8808a45d7054.png)

When we take two points in the latent space which will generate two different faces, we can create a transition or interpolation of the two faces by taking a linear path between the two points. Here is the iterpolation result shows the first images gradually transitioned to the second image. 
![屏幕截图 2021-06-30 201913](https://user-images.githubusercontent.com/70667153/124075770-f1586e80-da77-11eb-8647-85aafa5fc4da.png)

We finally try to make the interpolation animation using moviepy library to create a video. 

https://user-images.githubusercontent.com/70667153/124077423-1817a480-da7a-11eb-865a-d0be2bbee814.mp4

## StyleGAN-ada Implementation with PyTorch
Also we tired to generate real person on the Colab. But this time we try a new stuff called StyleGAN2-ada for pytorch which is released not that long ago. What’s really exciting about this new implementation of stylegan2 is the original stylegan2 and the ada variant of it used a very old version of tensorflow. That made it very difficult to use if we have the latest cuda 11drivers. This ada use pytorch rather than tensorflow. And it’s very easy to install. 

The detailed coding can be seen in the StyleGAN2-ADA-Pytorch.ipynb
[ipython notebook](https://github.com/xinyiff/Project-on-StyleGAN/blob/67697a04c808be8a21fa24536815c00d912f1512/StyleGAN2-ADA-Pytorch.ipynb)

https://user-images.githubusercontent.com/70667153/124163987-cf400a00-dad2-11eb-91fc-f5a0d2c7d374.mp4


# References

[1]. https://arxiv.org/abs/1812.04948
Tero Karras, Samuli Laine, Timo Aila, A Style-Based Generator Architecture for Generative Adversarial Networks, arXiv:1812.04948v3 [cs.NE] 29 Mar 2019

[2]. https://arxiv.org/abs/1710.10196
Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen, Progressive Growing of GANs for Improved Quality, Stability, and Variation, arXiv:1710.10196v3 [cs.NE] 26 Feb 2018

[3]. https://arxiv.org/abs/2006.06676
Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Training Generative Adversarial Networks with Limited Data, arXiv:2006.06676v2 [cs.CV] 7 Oct 2020

[4]. http://arxiv.org/abs/1912.04958
Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Analyzing and Improving the Image Quality of StyleGAN, arXiv:1912.04958v2 [cs.CV] 23 Mar 2020

[5]. Kevin Ashley, Make Art With Aritificial Intelligence, 2021, The Art of AI Collection
