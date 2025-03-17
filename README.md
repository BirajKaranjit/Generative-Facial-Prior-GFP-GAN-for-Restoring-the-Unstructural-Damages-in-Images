# Generative-Facial-Prior-GFP-GAN-for-Restoring-the-Unstructural-Damages-in-Images

Old Image Restoration is an innovative application of deep learning designed to revive aged
photographs by addressing damages such as scratches, tears, creases, and fading. This
work begins with scratch detection using a Swin U-Net architecture that combines the Swin
Transformer’s robust hierarchical feature extraction with the U-Net’s precise localization to
accurately segment scratch regions from undamaged areas.Building on this, the approach
employs image inpainting via Generative Adversarial Networks (GANs) to restore areas affected by both structural and unstructured damage. The AOT-GAN model is utilized for
its high-quality, context-aware completions, effectively reconstructing missing or degraded
regions to preserve important historical details.
To further enhance visual appeal, image colorization is performed using a ResNet-18 based
GAN integrated with perceptual loss functions, which revitalizes grayscale images with natural and vivid colors. 
Additionally, GFP-GAN is applied to mitigate unstructured damages such as blur, noise, and low resolution, ensuring an overall improvement in image quality.Comprehensive evaluations using metrics like Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) confirm significant enhancements in visual fidelity.
