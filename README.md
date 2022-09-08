# SynGAN
SynGAN: a generative adversarial network to generate synthetic 7T images from the widely used 3T images

# Background
7T MRI has the potential to improve the diagnosis and monitoring of diseases such as multiple sclerosis, cerebrovascular, brain tumors, and aging-related brain changes. However, 7T MRI scanners are much more expensive and not always available in the clinics. So far, there are less than 100 7T MRI scanners compared with more than 20,000 3T MRI scanners in the world. Therefore, synthesizing 7T MR images from the widely used 3T images is highly desirable for both clinical and research applications.

# Method 
SynGAN consists of a generator and a discriminator. The generator is used to synthesize 7T images from the corresponding 3T images, and the discriminator tries to distinguish the synthetic 7T images from the real ones. We adopt U-Net as the generator, because the U-Net has advantages of multilevel decomposition, multichannel filtering and multiscale skip connections.

# System requirement
 PyTorch package (version 1.09; https://pytorch.org)
 
 This code was adapted from a medical image-to-image translation (https://github.com/Kid-Liet/Reg-GAN) study for the application of synthesizing 7T images from 3T images.
