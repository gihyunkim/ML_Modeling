# ResNet50 or 101 with tensorflow 2.0  
datasets : cifar10  
optimization : Adam(Momentum in paper)  
loss : sparse categorical crossentropy  


# first try
- learning rate 0.1 -> 0.01(25 epochs) -> 0.001(50 epochs)
- no regularization term
- no augmentation
- ResNet 50
- batch size = 128

**overfitting : train acc = almost 100%, test acc = 84%**

# second try with image data generation
- image generator : rescale:1/255, rotation:20, width_shift:0.1, height_shift:0.1, validation_split:0.1
- learning rate 0.03 -> 0.003(25 epochs) -> 0.0003(50 epochs)
- batch_size = 64
- changed optimization to Momentum(0.9)
- much longer train time
- seem like overfitting has been reduced.
