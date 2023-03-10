# art.AI

**Team members:** Maika Nishiguchi, Seho Kwak, and Rachel Yang

![](cs152.jpeg)

### Introduction 
Transferring the style from one image to another is a fascinating area of research in computer vision and graphics that has been rapidly growing in popularity. It is a process that can be considered as a problem of texture transfer. In texture transfer, the goal is to synthesize a texture from a source image while constraining the texture synthesis in order to preserve the semantic content of a target image. Essentially, the aim of style transfer is to create an image that is a perfect blend of the content of one image and the style of another.
	
To achieve this, a style transfer algorithm should be able to extract the semantic image content from the target image (e.g., the objects and general scenery) and inform a texture transfer procedure to render the semantic content of the target image in the style of the source image. Specifically, the algorithm needs to (1) recognize objects in a target image and (2) recombine the objects and style of a style image.
	
To accomplish this task, researchers have employed various methods, with convolutional neural networks (CNNs) being the most popular. Gatys et al. (2016) used image representations from CNNs optimized for object recognition that transfer the reference style image onto the input target image to produce an output. Similarly, Li et al. (2021) explained how CNNs can be used in the processing of 2-D images, including object detection and image classification. Luan et al. (2017) built on this algorithm and augmented it further to satisfy more photorealistic style transfer by suppressing distortion. Furthermore, Kotovenko et al. (2019) introduced a content transformation module between the encoder and decoder to reduce extra deformations, additions, and deletions of content details. They also utilized similar content appearing in photographs and style samples to learn how style alters content details and generalize this to other class details.

To train our model, we will be using ArtBench-10 as the base dataset. ArtBench-10 is the first class-balanced, high-quality, cleanly annotated, and standardized dataset for benchmarking artwork generation, introduced by Liao et al. (2022). The dataset comprises 60,000 images of artwork from 10 distinctive artistic styles, with 5,000 training images and 1,000 testing images per style. ArtBench-10 has several advantages over previous artwork datasets. Firstly, it is class-balanced while most previous artwork datasets suffer from the long tail class distributions. Secondly, the images are of high quality with clean annotations. Thirdly, ArtBench-10 is created with standardized data collection, annotation, filtering, and preprocessing procedures.
	
Our goal is to generate high-quality images in the style of the desired artist or artwork. We hope our results will support the conclusion that neural networks can be trained to generate new images in a specific style and have potential applications in areas such as digital art and advertising.
	
In summary, style transfer is a fascinating area of research that has been receiving increased attention in recent years. By leveraging convolutional neural networks and carefully curated datasets like ArtBench-10, we can produce high-quality images in a specific style that can have many practical applications in the real world. We look forward to seeing how this technology evolves and contributes to the field of computer vision and graphics.
	
### References
- Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2414-2423).
- Li, Z., Liu, F., Yang, W., Peng, S., & Zhou, J. (2021). A survey of convolutional neural networks: analysis, applications, and prospects. IEEE transactions on neural networks and learning systems.
- Luan, F., Paris, S., Shechtman, E., & Bala, K. (2017). Deep photo style transfer. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4990-4998).
- Kotovenko, D., Sanakoyeu, A., Ma, P., Lang, S., & Ommer, B. (2019). A content transformation block for image style transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10032-10041).
- Liao, P., Li, X., Liu, X., & Keutzer, K. (2022). The artbench dataset: Benchmarking generative models with artworks. arXiv preprint arXiv:2206.11404.
