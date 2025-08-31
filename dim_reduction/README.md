# Dimensionality Reduction

## Representation learning
General family. Goal: represent raw data in another space.  
Examples: PCA, ICA, autoencoders, embeddings, latent variable models, contrastive learning.  

## Dimensionality reduction
Subfamily of representation learning. Goal: reduce number of features, keep information.  
- Linear: PCA, LDA, ICA.  
- Nonlinear: Kernel PCA, t-SNE, UMAP, Isomap, autoencoders (nonlinear).  

## PCA
Principal Component Analysis. Linear, unsupervised.  
- Finds new axes (principal components) aligned with maximum variance.  
- Uses eigenvectors of covariance matrix.  
- Projects high-dim data into smaller space (k dimensions).  

### Why PCA
- Compression.  
- Noise reduction.  
- Visualization.  
- Preprocessing for ML.  
