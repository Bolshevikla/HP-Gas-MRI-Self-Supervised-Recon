# HP-Gas-MRI-Self-Supervised-Recon
Bayesian Inference-Based Self-Supervised Learning with Adaptive k-Space Division for Robust hyperpolarized Xe-129 pulmonary MRI Reconstruction

# Sampling distribution
![图片](https://github.com/user-attachments/assets/b67ba293-ddb2-4302-b4e8-0b61b3d736bc)

To analyze the spatial behavior of different k-space splitting strategies, we examined the normalized sampling distributions (Fig. X). The Gaussian strategy concentrates sampling in low-frequency regions, resulting in limited high-frequency coverage and lower entropy (3.11). Uniform sampling achieves the highest entropy (3.23) by distributing points evenly, but lacks task-specific adaptivity.

Our adaptive method maintains comparable entropy (3.22), indicating balanced coverage, but importantly increases sampling in the mid-to-high frequency range (normalized distance 0.4–0.8). This region is critical for preserving fine structural details in hyperpolarized 129129Xe MRI, where high-frequency information is easily lost due to rapid signal decay. These results show that our method effectively balances uniformity and adaptivity, learning to allocate samples where they are most needed for accurate reconstruction.

# Ventilation defect percentage
To assess the potential clinical utility of the reconstructed images for lung function evaluation, ventilation defects were segmented from the pulmonary gas MR images, and the ventilation defect percentage (VDP) was calculated. In the segmentation results, red regions indicate ventilation defects, while green regions represent well-ventilated (normal) lung areas.
<img width="2296" height="620" alt="VDP" src="https://github.com/user-attachments/assets/7e8d373d-5dfb-4227-9ff1-99cf97cc4e21" />
