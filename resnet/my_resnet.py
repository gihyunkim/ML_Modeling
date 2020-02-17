import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, EarlyStopping

epochs = 120
batch_num = 128
init_lr = 0.1
lr_list = [(0.1, 25), (0.01, 50), (0.001, 100)]
resnet_layers = 50


def lr_schedule(epoch):
    lr = init_lr
    for mul, start_epoch in lr_list:
        if epoch >= start_epoch:
            lr = lr * mul
        else:
            break
        # tf.summart.scalar('learning rate', data=lr, step=epoch)

    return lr


def get_residual_list(res_layers):
    if res_layers == 50:
        res_seq = [3, 4, 6, 3]
    elif res_layers == 101:
        res_seq = [3, 8, 23, 3]

    return res_seq


class myResnet:
    def __init__(self):
        pass

    # load cifar 10 data
    def load_datasets(self):
        (x_train, t_train), (x_test, t_test) = load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        datasets = [x_train, t_train, x_test, t_test]
        return datasets

    # residual block
    def residual_block(self, init_x, ch, downsample=False):
        # BN
        x = layers.BatchNormalization()(init_x)
        x = tf.nn.relu(x)

        if downsample is True:
            x = layers.Conv2D(filters=ch, kernel_size=3, strides=2, padding='same')(x)
            init_x = layers.Conv2D(filters=ch, kernel_size=1, strides=2, padding='same')(init_x)
        else:
            x = layers.Conv2D(filters=ch, kernel_size=3, strides=1, padding='same')(x)

        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        return x + init_x

    def bottleneck_residual_block(self, init_x, ch, downsample=False):
        # BN
        x = layers.BatchNormalization()(init_x)
        shortcut = tf.nn.relu(x)

        # 1x1 convolution
        x = layers.Conv2D(filters=ch, kernel_size=3, padding='same')(shortcut)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        if downsample is True:
            x = layers.Conv2D(filters=ch, kernel_size=3, strides=2, padding='same')(x)
            # init_x = layers.Conv2D(filters=ch, kernel_size=1, strides=2,padding='same')(init_x)
            shortcut = layers.Conv2D(filters=ch * 4, kernel_size=1, strides=2, padding='same')(shortcut)
        else:
            x = layers.Conv2D(filters=ch, kernel_size=3, strides=1, padding='same')(x)
            shortcut = layers.Conv2D(filters=ch * 4, kernel_size=1, strides=1, padding='same')(shortcut)

        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = layers.Conv2D(filters=ch * 4, kernel_size=1, strides=1, padding='same')(x)

        return x + shortcut

    # ResNet 50 layer
    def create_network(self, res_layer):

        # input shape
        inputs = layers.Input(shape=(32, 32, 3))

        init_ch = 32

        res_seq = get_residual_list(res_layer)

        if res_layer < 50:
            resblock = self.residual_block
        else:
            resblock = self.bottleneck_residual_block

        # 3x3 64
        x = layers.Conv2D(filters=init_ch, kernel_size=3)(inputs)

        # 64ch residual
        for _ in range(res_seq[0]):
            x = resblock(x, init_ch)

        # 128
        x = resblock(x, init_ch * 2, downsample=True)
        for _ in range(res_seq[1]):
            x = resblock(x, init_ch * 2)

        # 256
        x = resblock(x, init_ch * 4, downsample=True)
        for _ in range(res_seq[2]):
            x = resblock(x, init_ch * 4)

        # 512
        x = resblock(x, init_ch * 8, downsample=True)
        for _ in range(res_seq[3]):
            x = resblock(x, init_ch * 8)

        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(10, activation='softmax')(x)
        Model = models.Model(inputs=inputs, outputs=x)
        return Model


# tf.keras.backend.set_floatx('float32')
resnet = myResnet()
datasets = resnet.load_datasets()
x_train, t_train = datasets[0], datasets[1]
x_test, t_test = datasets[2], datasets[3]
# x_train = tf.image.resize(x_train,(8,8))
# x_test = tf.image.resize(x_test,(8,8))

# callback func
lr_schedule_callback = LearningRateScheduler(lr_schedule)

'''
tensorboard_callback = TensorBoard(
    update_freq='batch', histogram_freq=1
)
'''

# early_callback = EarlyStopping(monitor='val_loss',patience=3)

model = resnet.create_network(resnet_layers)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr),
    metrics=['sparse_categorical_accuracy']
)

model.fit(x_train, t_train,
          batch_size=batch_num, epochs=epochs, shuffle=True,
          validation_data=(x_test, t_test), steps_per_epoch=x_train.shape[0] // batch_num,
          callbacks=[lr_schedule_callback]
          )

test_loss, test_acc = model.evaluate(x_test, t_test, verbose=2)
print(test_acc)