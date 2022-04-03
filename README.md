# Inception-ResNet-V2 : Face Recognition

Reference Paper: [Inception-v4, Inception-ResNet and
the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261v1.pdf)

## Design
- Written in PyTorch, this model has a total of **26 high-level blocks** and can classify upto **1001** different classes of images. 
- It has a complete depth of 164 layers 
- **Input-size** = `3 x 299 x 299`

## Implementation details
### Model design
  - The individual blocks have been defined seperately with explicit mention of `in_channels` and `out_channels` for each layer to maintain a visual flow of how images are moving around in the network
  - A custom `LambdaScale` layer has been introduced to scale the residuals, as discussed in the original paper to tackle the problem of the later layers dying early in training
  - Batch Normalization has been done to ensure regularization 
  ![image](https://user-images.githubusercontent.com/70141886/161433653-8d46d38e-39ab-4bc0-b335-ef374b45b469.png)
  <p align=center> Layer design - Overview </p>

### Parameters
  - Loss function : `torch.nn.CrossEntropyLoss()`
  - Optimizer : `torch.optim.Adam(amsgrad=True)`
  - Scheduler : `torch.optim.lr_scheduler.ReduceLROnPlateau(mode='min', factor=0.2, threshold=0.01, patience=5)`

### Training
  - `prefetch_generator.BackgroundGenerator` has been used to bring about computational efficiency by pre-loading the next mini-batch during training
  - The `state_dict` of each epoch is stored in the `resnet-v2-epochs` directory (created if does not exist)
  - By default, it will try to **run training using a CUDA GPU**, but it will back up to a CPU on not being able to detect the presence of one
  - Parallelization has not been implemented as a design choice in order to keep the training function readable and easy to implement
  - The results of the training session can be **viewed interactively** using `TensorBoard` with logs being stored in `/runs` directory
  - A benchmark of 00:30:03 minutes was seen on a NVIDIA GTX 1650Ti 4GB, Intel i7-10750H, 16GB RAM, SSD-enabled computer to train 1 epoch

### Dataset and Pre-processing details
  - Using the Face-Expression-Recognition-Dataset from `jonathanoheix` on Kaggle, we train on a total of 28,821 images of 7 different classes including 'Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad' and 'Surprise'
  - We perform some simple preprocessing using `torchvision.transforms.Resize()`, `torchvision.transforms.ToTensor()` and `torchvision.transforms.Normalize()` to get a good representation for the images in tensor format.
  - We make use of `torch.utils.data.DataLoader()` to improve load times and process images in random mini-batches in an efficient and optimized manner

## Credits
- **Paper**: *Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning*
- **Authors**: *Christian Szegedy, Sergey Ioffe and Vincent Vanhoucke*
- **Images dataset - Source** : [Face Expression Recognition Dataset](https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset)
