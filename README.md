# LsmGANs: Image-domain least-squares migration using a new framework of generative adversarial networks

We develop a new GAN framework that is more suitable for the Hessian approximation of image-domain LSM, which is named as LsmGANs. In the new framework, we use a max-pooling instead of convolution to downsample the feature maps to capture horizontal and vertical variations of reflectors. This enables us to map reflection events to correct location in downsampling. To address the lateral discontinuity of events in the predicted image from conventional GANs, we further apply multiple transform layers to strengthen feature transformation to guide Hessian approximation. Finally, we add the skip connection in the transform layer to enhance the information exchange of the feature channels and avoid the gradient vanishing problem to improve image resolution. Below are the relevant figures and the full code.

![b09c5f748e59d0e3f31285fe81437c1](README.assets/b09c5f748e59d0e3f31285fe81437c1.png)

![ef934d1320d97e64ca02d9cf1e09d3c](README.assets/ef934d1320d97e64ca02d9cf1e09d3c.png)

![dae2ca03b0305af4472e8285128d7b5](README.assets/dae2ca03b0305af4472e8285128d7b5.png)



## Link to related article

[click this link](https://ieeexplore.ieee.org/abstract/document/10215494) for more details.



## Publications

When using this model, please cite the following references:

**GB/T 7714:**

```
J. Sun, J. Yang, J. Huang, Y. Yu, Z. Li and C. Zhao, "LsmGANs: Image-Domain Least-Squares Migration Using a New Framework of Generative Adversarial Networks," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-13, 2023, Art no. 5916913, doi: 10.1109/TGRS.2023.3304726.
```

**Bib Tex:**

```
@ARTICLE{10215494,
  author={Sun, Jiaxing and Yang, Jidong and Huang, Jianping and Yu, Youcai and Li, Zhenchun and Zhao, Chong},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={LsmGANs: Image-Domain Least-Squares Migration Using a New Framework of Generative Adversarial Networks}, 
  year={2023},
  volume={61},
  number={},
  pages={1-13},
  doi={10.1109/TGRS.2023.3304726}}
```
