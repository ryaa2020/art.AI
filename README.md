# art.AI

**Team members:** Maika Nishiguchi, Seho Kwak, and Rachel Yang

![](cs152.jpeg)

### Introduction & Related Works
Neural style transfer is a captivating and rapidly expanding area of research in computer vision and graphics. This technique involves transferring the style from one image to another, essentially blending the content of a target image with the style of a source image. The process can be viewed as a texture transfer problem, where the goal is to synthesize a texture from the source image while preserving the semantic content of the target image.

A successful style transfer algorithm should extract the semantic content of the target image, such as objects and general scenery, and use this information to guide a texture transfer process that renders the content in the style of the source image. This requires the algorithm to (1) recognize objects in the target image and (2) recombine the objects and style of the source image.

Convolutional neural networks (CNNs) have emerged as the most popular method for achieving this task. Gatys et al. (2016) demonstrated the use of image representations from CNNs optimized for object recognition to transfer the style of a reference image onto an input target image. Similarly, Li et al. (2021) explored how CNNs can be applied to process 2-D images, including object detection and image classification. Luan et al. (2017) built upon this approach by augmenting the algorithm to achieve more photorealistic style transfer while minimizing distortion. Additionally, Kotovenko et al. (2019) introduced a content transformation module between the encoder and decoder to reduce extra deformations, additions, and deletions of content details, learning how style influences content details and generalizing this to other class details.

For our project, we will use baby photos from the dataset https://www.kaggle.com/datasets/cocowaffle/baby-photos as content images and famous artworks from https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time as style images. We will start by replicating the neural style transfer tutorial available at https://pytorch.org/tutorials/advanced/neural_style_tutorial.html and adjust the parameters to train our model using the chosen datasets. Our primary focus will be on baby photos, applying different art styles to these images.

Our goal is to split one photo in half, applying two different art styles to each half of the image. We will experiment with blending multiple styles into the output image and apply each style to different regions of the image.

### Methods Outline
1. We will employ PyTorch as the primary software for our neural network implementation, taking advantage of its flexibility and efficiency in building and training models.
2. We will use the baby photos dataset (https://www.kaggle.com/datasets/cocowaffle/baby-photos) as our content images and the best artworks dataset (https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) for style images.
3. A convolutional neural network (CNN) will be the core of our model, as it is adept at identifying patterns and features in images, which is crucial for recognizing different art styles.
4. Our inputs will consist of three-channel images represented as matrices of pixel values, which the model will use to extract content and style information.
5. The output of our neural style transfer model will be a visually appealing image that fuses the style of two different images with the content of a baby photo. This three-channel image, with the same shape and type as the input image, is intended to produce an aesthetic result rather than serve any specific classification, regression, or segmentation task.

### Discussion Section Outline:
1. We will present the results of our neural style transfer model applied to baby photos, showcasing the visual appeal and effectiveness of combining two different art styles in one image. (Add example images for visual reference.)
2. To evaluate our model’s performance, we will assess the quality of the style transfer, the preservation of content, and the blending of the two styles in the output images. (Discuss evaluation metrics.)
3. We will compare our work to the original neural style transfer algorithm and other variations, highlighting any improvements or unique features in our approach. (Add specific comparisons to other papers or implementations.)
4. Our project builds upon the PyTorch tutorial and demonstrates the potential of using custom datasets to create unique applications of neural style transfer, such as blending multiple styles in baby photos. (Discuss the importance of exploring diverse applications.)
5. We will provide a tutorial-style explanation of the key concepts and techniques we have learned and implemented, including the convolutional neural network architecture and the process of splitting and blending styles in an image. (Include a step-by-step guide or code snippets.)
As we finalize our results and analysis, we will expand upon these points and include additional insights, visualizations, and comparisons to support our claims and illustrate the potential of our approach in the context of neural style transfer applications.

### References
- Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2414-2423).
- Li, Z., Liu, F., Yang, W., Peng, S., & Zhou, J. (2021). A survey of convolutional neural networks: analysis, applications, and prospects. IEEE transactions on neural networks and learning systems.
- Luan, F., Paris, S., Shechtman, E., & Bala, K. (2017). Deep photo style transfer. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4990-4998).
- Kotovenko, D., Sanakoyeu, A., Ma, P., Lang, S., & Ommer, B. (2019). A content transformation block for image style transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10032-10041).
- Liao, P., Li, X., Liu, X., & Keutzer, K. (2022). The artbench dataset: Benchmarking generative models with artworks. arXiv preprint arXiv:2206.11404.
