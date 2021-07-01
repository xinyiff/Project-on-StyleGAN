# Generating Images with StyleGAN

In this project, we are going to introduce the basic idea of StyleGAN. Generative Adversarial Networks, or GANs for short, have appeared frequently in the media, which showcasing their ability to generate large high-quality images. In 2018, NVDIA introduce a Style Generative Adersarial Network, or StyleGAN for short, which was one significant step forward for realistic face generation. Next further, StyleGAN was followed by StyleGAN2 in 2019, which improved the quality of StyleGAN by removing certain srtifacts. For the experiment, we will see both how to train StyleGAN2 on any arbitrary set of images; as well as use different pretained weights on Google Colab and local enviroment.

## StyleGAN Architecture 

## StyleGAN2 Architecture

Based on the StyleGAN, researchers led by Tero Karras have published a paper where they analyze the capabilities of the original StyleGAN architecture and propose a new improved version -- StyleGAN2. They propose several modifications both in architecture and the training strategy.

<img width="676" alt="Screen Shot 2021-06-29 at 4 48 15 PM" src="https://user-images.githubusercontent.com/70667153/124072882-f5828d00-da73-11eb-9cb5-7b0d8a7f502d.png">

In particular, researchers redesigned the generator normalization, they revisited the progressive growing as the training stabilization and introduced a new regularization technique to improve conditional generation. The generator architecture was modified such that AdaIn layers were removed i.e adaptive instance normalization was replaced with a “demodulation” operation. After inspecting the effects of progressive growing as a procedure for training with large resolution images, researchers propose an alternative approach where training starts by focusing on low-resolution images and then progressively shifts focus to higher and higher resolutions but without changing the network topology.


## Training Method



## StyleGAN Implementation with TensorFlow

### Generating Anime Characters on Google Colab

### Generating Real Person Images on Windows 10

## StyleGAN-ada Implementation with PyTorch

## References
