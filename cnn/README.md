# my cnn implementation
## environment
- tensorflow 1.13.2
- learning rate 1e-4

## using callback
- learning rate decay
- early stopping

### phase 1
conv1 -> pool1 -> conv2 -> pool2 -> conv3 -> conv4 -> conv5 -> fc1 -> fc2

accuracy : 87%

### phase 2
conv1 -> bn1 -> pool1 -> conv2 -> bn2 ->pool2 -> conv3 -> bn3 -> conv4 -> bn4 -> fc1 -> bn_fc1 -> fc2 -> bn_fc2

accuracy : 13%

**\* phase 1 showed overfitting so that I puts batch normalization, but it makes worse to overfitting.**
