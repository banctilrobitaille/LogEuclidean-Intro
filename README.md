#  <img src="/icons/icon.png" width="90" vertical-align="bottom">Log-Euclidean and DTI Introduction
> This repository contains resources that helps in the understanding of Diffusion Tensor Imaging processing using the
Log-Euclidean framework. It is recommended to read the proposed articles in order to fully understand the Jupyter Notebook.

## Articles
### Diffusion Tensor Imaging Basics
 - Mori, S., & Zhang, J. (2006). Principles of Diffusion Tensor Imaging and Its Applications to Basic Neuroscience Research.
 - Maier-Hein, K. H., Neher, P. F., Houde, J.-C., Côté, M.-A., Garyfallidis, E., Zhong, J., … Descoteaux, M. (2017). The challenge of mapping the human connectome based on diffusion tractography. Nature Communications
### Log-Euclidean Framework
 - Arsigny, V., Fillard, P., Pennec, X., & Ayache, N. (2006). Log-Euclidean metrics for fast and simple calculus on diffusion tensors. Magnetic Resonance in Medicine
 - Arsigny, V., Fillard, P., Pennec, X., & Ayache, N. (2006). Geometric means in a novel vector space structure on symmetric positive-definite matrices. SIAM Journal on Matrix Analysis and Applications
### Generative Adversarial Networks (GANs)
 - Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., … Bengio, Y. (2014). Generative adversarial nets. In Advances in Neural Information Processing Systems. Neural information processing systems foundation.
 - Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN.
 - Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks. Proceedings of the IEEE International Conference on Computer Vision
 - Yi, X., Walia, E., & Babyn, P. (2019). Generative adversarial network in medical imaging: A review. Medical Image Analysis
### Deep Neural Networks for SPD Matrices Processing and DTI Sysnthesis
 - Huang, Z., & Van Gool, L. (2017). A riemannian network for SPD matrix learning. In 31st AAAI Conference on Artificial Intelligence, AAAI 2017
 - Huang, Z., Wu, J., & Van Gool, L. (2019). Manifold-Valued Image Generation with Wasserstein Generative Adversarial Nets. Proceedings of the AAAI Conference on Artificial Intelligence
 - Gu, X., Knutsson, H., Nilsson, M., & Eklund, A. (2019). Generating Diffusion MRI Scalar Maps from T1 Weighted Images Using Generative Adversarial Networks. Lecture Notes in Computer Science
 - Zhong, J., Wang, Y., Li, J., Xue, X., Liu, S., Wang, M., … Li, X. (2020). Inter-site harmonization based on dual generative adversarial networks for diffusion tensor imaging: Application to neonatal white matter development. BioMedical Engineering Online
### Advanced Topics
 - Ionescu, C., Vantzos, O., & Sminchisescu, C. (2015). Matrix backpropagation for deep networks with structured layers. In Proceedings of the IEEE International Conference on Computer Vision
 - Brooks, D., Schwander, O., Barbaresco, F., Schneider, J.-Y., & Cord, M. (2019). Riemannian batch normalization for SPD neural networks.


## Log-Euclidean Introduction Notebook
1. Clone the project
    - git clone https://github.com/banctilrobitaille/LogEuclidean-Intro.git
2. Setup a virtual environment
    - For Pycharm: https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html
3. Install the dependencies: 
    - Install Pytorch from https://pytorch.org/
    - Install other libraries: pip install -r requirements.txt
4. Install a Jupyter kernel
    - ipython kernel install --user --name=.venv
    - See: https://medium.com/@eleroy/jupyter-notebook-in-a-virtual-environment-virtualenv-8f3c3448247
5. Run Jupyter and select the provided Jupyter notebook
    - jupyter notebook
    - Select LogEuclideanIntro.ipynb and make sure that the selected kernel is : .venv