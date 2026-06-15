# Progress Report Draft Outline

## Brief Project Description

This project generates 64x64 RGB anime face images using generative adversarial networks. The input to the model is a random latent vector, and the output is a synthetic anime-style face. The project is interesting because the result is visual and easy to inspect, while still requiring the model to learn complex image patterns such as face shape, eyes, hair, colour, and style. Deep learning is appropriate because these patterns are difficult to describe manually, and GANs are designed to learn an image distribution and generate new samples from it.

## Data Processing

Data source: Kaggle Anime Faces dataset by Soumik Rakshit. Images are recursively loaded from the raw dataset folder, unreadable/corrupted files are skipped, images are converted to RGB, center-cropped to square, resized to 64x64, and saved as PNG files in `data/processed/anime_faces_64`. Dataset statistics and a cleaned sample grid are saved under `runs/data_prep`.

For final testing on never-before-seen data, the plan is to collect a small separate anime face set from sources not used in training, or manually hold out a separate external dataset. This avoids judging the final generator only on samples from the same distribution used during development.

## Baseline Model: DCGAN

The baseline model is a standard DCGAN with a transposed-convolution generator and convolutional discriminator. DCGAN is a reasonable baseline because it is simple, well-known, and directly designed for 64x64 image generation. Quantitative evidence will include generator and discriminator loss curves. Qualitative evidence will include generated sample grids at different epochs.

## Primary Model: WGAN-GP

The primary model is WGAN-GP, which keeps the same general convolutional generator/critic idea but changes the training objective to Wasserstein loss with gradient penalty. This choice makes sense because GAN training can be unstable, and WGAN-GP is designed to reduce instability and mode collapse. Quantitative evidence will include critic/generator loss and gradient penalty trends. Qualitative evidence will compare generated images against the DCGAN samples.

## Challenges

The main challenges are GAN training instability, slow training without strong GPU compute, possible mode collapse, and uncertainty in dataset provenance/licensing. The progress report can frame early blurry/noisy samples as expected for short training, as long as the pipeline works end-to-end and the report clearly explains the next steps.
