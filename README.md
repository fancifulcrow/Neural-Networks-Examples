# Neural Networks Examples
 
This repository contains implementations of 3 Supervised and 3 Unsupervised Neural Networks.

- **Supervised Models**
    - Artificial Neural Networks
    - Convolutional Neural Networks
    - Recurrent Neural Networks
- **Unsupervised Models**
    - Self-Organizing Maps
    - Boltzmann Machines
    - Autoencoders

## Neural Networks Overview

### Artificial Neural Network
Artificial Neural Networks (ANNs) are the foundation of deep learning. They consist of interconnected nodes organized in layers, with each node performing a simple computation. ANNs are commonly used for regression and classification tasks.

### Convolutional Neural Network
Convolutional Neural Networks (CNNs) are specialized for processing structured grid data, such as images. They employ convolutional layers to automatically learn hierarchical patterns and are widely used in image classification, object detection, and segmentation tasks.

### Recurrent Neural Network
Recurrent Neural Networks (RNNs) are designed to handle sequential data with dependencies over time. They have loops that allow information to persist, making them suitable for tasks like sequence generation, machine translation, and time series prediction.

### Self-Organizing Map
Self-Organizing Maps (SOMs) are unsupervised neural networks used for dimensionality reduction and clustering. They organize high-dimensional data into a low-dimensional grid while preserving the topological properties of the input space.

### Boltzmann Machine
More Specifically, the Restricted Boltzmann Machine. Restricted Boltzmann Machines (RBMs) are probabilistic generative models that learn a probability distribution over its set of inputs. They are building blocks for deeper models such as Deep Belief Networks and are used in collaborative filtering, feature learning, and more.

### Autoencoder
In this case, Variational Autoencoders. Variational Autoencoders (VAEs) are generative models that learn to represent high-dimensional data in a low-dimensional latent space. They enable efficient generation of new data samples and are used in applications like image generation, anomaly detection, and semi-supervised learning.

## Dependencies
The implementations are written in Python and require the following dependencies:
- Tensorflow
- Pytorch
- Scikit-learn
- Numpy
- Pandas
- Matplotlib
- [MiniSom](https://github.com/JustGlowing/minisom)

## Additional Reading

If you're interested in learning more about neural networks and their applications, here are some recommended resources:

### Papers

- [Y. Lecun, L. Bottou, G. Orr, and K.-R. Müller, “Efficient BackProp,” Aug. 2000.](https://www.researchgate.net/publication/2811922_Efficient_BackProp)

- [A. Karpathy, J. Johnson, and L. Fei-Fei, “Visualizing and Understanding Recurrent Networks,” 2015"](https://arxiv.org/pdf/1506.02078.pdf)

- [J. Wu, “Introduction to convolutional neural networks,” 2017](https://cs.nju.edu.cn/wujx/paper/CNN.pdf)

- [T. Kohonen, “The self-organizing map,” 1990](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1990-Kohonen-PIEEE.pdf)

- [A. Fischer and C. Igel, “Training restricted Boltzmann machines: An introduction,” 2014](https://christian-igel.github.io/paper/TRBMAI.pdf)

- [U. Michelucci, "An Introduction to Autoencoders," 2022.](https://arxiv.org/pdf/2201.03898.pdf)

### Blogs
- [Introduction to Neural Networks by Matthew Stewart, 2019](https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c)

- [The Ultimate Guide to Convolutional Neural Networks (CNN) by SuperDataScience Team, 2018](https://www.superdatascience.com/blogs/the-ultimate-guide-to-convolutional-neural-networks-cnn)

- [Understanding LSTM Networks by Christopher Olah, 2015](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

- [The Unreasonable Effectiveness of Recurrent Neural Networks by Andrej Karpathy, 2015](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

- [The Ultimate Guide to Self Organizing Maps (SOM's) by SuperDataScience Team, 2018](https://www.superdatascience.com/blogs/the-ultimate-guide-to-self-organizing-maps-soms)

- [Beginner's Guide to Boltzmann Machines in PyTorch](https://blog.paperspace.com/beginners-guide-to-boltzmann-machines-pytorch/)

- [Neural Networks Are Impressively Good At Compression by Malte Skarupke, 2016](https://probablydance.com/2016/04/30/neural-networks-are-impressively-good-at-compression/)

- [Introduction To Autoencoders by Abhijit Roy, 2020](https://towardsdatascience.com/introduction-to-autoencoders-7a47cf4ef14b)

- [Variational AutoEncoders (VAE) with PyTorch by Alexander Van de Kleut, 2020](https://avandekleut.github.io/vae/)