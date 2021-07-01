# Generating Images with StyleGAN

In this project, we are going to introduce the basic idea of StyleGAN. Generative Adversarial Networks, or GANs for short, have appeared frequently in the media, which showcasing their ability to generate large high-quality images. In 2018, NVDIA introduce a Style Generative Adersarial Network, or StyleGAN for short, which was one significant step forward for realistic face generation. Next further, StyleGAN was followed by StyleGAN2 in 2019, which improved the quality of StyleGAN by removing certain srtifacts. For the experiment, we will see both how to train StyleGAN2 on any arbitrary set of images; as well as use different pretained weights on Google Colab and local enviroment.

## StyleGAN Architecture 

## StyleGAN2 Architecture

Based on the StyleGAN, researchers led by Tero Karras have published a paper where they analyze the capabilities of the original StyleGAN architecture and propose a new improved version -- StyleGAN2. They propose several modifications both in architecture and the training strategy.

<img width="729" alt="Screen Shot 2021-06-28 at 9 43 35 PM" src="https://user-images.githubusercontent.com/70667153/124073815-20211580-da75-11eb-92a9-3b3ee57e4cc2.png">
<img width="676" alt="Screen Shot 2021-06-29 at 4 48 15 PM" src="https://user-images.githubusercontent.com/70667153/124072882-f5828d00-da73-11eb-9cb5-7b0d8a7f502d.png">

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

As the figure shows, progressive growing leads to “phase” artifacts. In this example the teeth do not follow the pose but stay aligned to the camera, as indicated by the blue line. So the researchers use a hierarchical generator with skip connection (similar to MSG-GAN) instead of progressive growing. In this way, phase artifacts are reduced.

## StyleGAN Implementation with TensorFlow

### Requirements
- Both Linux and Windows are supported. Linux is recommended for performance and compatibility reasons.
- 64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.14.3 or newer.
- We recommend TensorFlow 1.14, which we used for all experiments in the paper, but TensorFlow 1.15 is also supported on Linux. TensorFlow 2.x is not supported.
- On Windows you need to use TensorFlow 1.14, as the standard 1.15 installation does not include necessary C++ headers.
- One or more high-end NVIDIA GPUs, NVIDIA drivers, CUDA 10.0 toolkit and cuDNN 7.5. To reproduce the results reported in the paper, you need an NVIDIA GPU with at least 16 GB of DRAM.
- Docker users: use the provided Dockerfile to build an image with the required library dependencies.

### Generating Anime Characters on Google Colab

In this part, we use the pre-trained Anime StyleGAN2 by Aaron Gokslan so that we can directly load the model and generate the anime images. For the implement convience, we would like to coding on the Google Colab this time. The detailed coding can be seen in the StyleGAN2_Animation.ipynb. 
[ipython notebook](https://github.com/xinyiff/Project-on-StyleGAN/blob/ee4e3ffde5707fabee05f5efc85a66bc26a44d33/StyleGAN2_Animation.ipynb)


![屏幕截图 2021-06-30 201944](https://user-images.githubusercontent.com/70667153/124075687-da198100-da77-11eb-94f9-57401a3597f9.png)
![屏幕截图 2021-06-30 201827](https://user-images.githubusercontent.com/70667153/124075757-ee5d7e00-da77-11eb-95f1-8808a45d7054.png)
![屏幕截图 2021-06-30 201913](https://user-images.githubusercontent.com/70667153/124075770-f1586e80-da77-11eb-8647-85aafa5fc4da.png)

### Generating Real Person Images on Windows 10

## StyleGAN-ada Implementation with PyTorch

## References
