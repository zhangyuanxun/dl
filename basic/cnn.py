import tensorflow as tf
import numpy as np


class CNN(object):
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.weights = {
            'conv1': tf.Variable(tf.random.normal([3, 3, 1, 64], stddev=0.1)),
            'conv2': tf.Variable(tf.random.normal([3, 3, 64, 128], stddev=0.1)),
            'fc1': tf.Variable(tf.random.normal([7 * 7 * 128, 1024], stddev=0.1)),
            'fc2': tf.Variable(tf.random.normal([1024, self.num_outputs], stddev=0.1)),
        }

        self.biases = {
            'bc1': tf.Variable(tf.random.normal([64], stddev=0.1)),
            'bc2': tf.Variable(tf.random.normal([128], stddev=0.1)),
            'bf1': tf.Variable(tf.random.normal([1024], stddev=0.1)),
            'bf2': tf.Variable(tf.random.normal([self.num_outputs], stddev=0.1)),
        }

    def __call__(self, X, keep_prob=1.0):
        # conv layer 1
        conv1 = tf.nn.conv2d(X, self.weights['conv1'], strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, self.biases['bc1']))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, rate=1-keep_prob)

        # conv layer 2
        conv2 = tf.nn.conv2d(conv1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2, self.biases['bc2']))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, rate=1-keep_prob)

        # fc layer 1
        conv2_shape = conv2.get_shape().as_list()
        dense1 = tf.reshape(conv2, shape=[-1, conv2_shape[1] * conv2_shape[2] * conv2_shape[3]])
        fc1 = tf.nn.relu(tf.add(tf.matmul(dense1, self.weights['fc1']), self.biases['bf1']))
        fc1 = tf.nn.dropout(fc1, rate=1-keep_prob)

        # out layer (fc layer 2)
        out = tf.add(tf.matmul(fc1, self.weights['fc2']), self.biases['bf2'])
        return out

    def trainable_variables(self):
        return [self.weights['conv1'], self.weights['conv2'], self.weights['fc1'], self.weights['fc2'],
                self.biases['bc1'], self.biases['bc2'], self.biases['bf1'], self.biases['bf2']]


def loss_fn(Y, Y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, Y_pred))


def mnist_datasets():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # # original size 60000*28*28 -> 60000 * 60000*28*28*1
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # normalize dataset
    x_train, x_test = x_train / np.float32(255), x_test / np.float32(255)
    y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)

    # one-hot encoding
    y_train, y_test = np.eye(10)[y_train], np.eye(10)[y_test]
    y_train, y_test = y_train.astype('float32'), y_test.astype('float32')
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_dataset, test_dataset


def train_epoch(model, optimizer, dataset, epoch_loss, epoch_acc, keep_prob):
    for (batch, (inputs, labels)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            preds = model(inputs, keep_prob)
            loss = loss_fn(labels, preds)

        grads = tape.gradient(loss, model.trainable_variables())
        optimizer.apply_gradients(zip(grads, model.trainable_variables()))

        # update metrics
        epoch_loss.update_state(loss)
        epoch_acc.update_state(labels, preds)


def train_fn(model, optimizer, num_epoch, train_dataset):
    display_step = 5
    keep_prob = 0.5
    for epoch in range(num_epoch):
        epoch_loss = tf.keras.metrics.Mean()
        epoch_acc = tf.keras.metrics.CategoricalAccuracy()

        train_epoch(model, optimizer, train_dataset, epoch_loss, epoch_acc, keep_prob)

        if epoch % display_step == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Training accuracy: {:.3%}".format(epoch + 1,
                                                                        epoch_loss.result(),
                                                                        epoch_acc.result()))


def test_fn(model, dataset):
    test_acc = tf.keras.metrics.CategoricalAccuracy()

    for batch, (inputs, labels) in enumerate(dataset):
        preds = model(inputs)
        test_acc.update_state(labels, preds)

    print("Test set accuracy: {:.3%}".format(test_acc.result()))


if __name__ == '__main__':
    num_epoch = 20
    batch_size = 100
    num_train = 60000
    num_test = 10000
    num_inputs = 784
    num_outputs = 10

    # load mnist datasets
    train_dataset, test_dataset = mnist_datasets()
    train_dataset = train_dataset.shuffle(num_train).batch(batch_size=batch_size)
    test_dataset = test_dataset.batch(batch_size=batch_size)

    model = CNN(num_inputs=num_inputs, num_outputs=num_outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # train the model
    train_fn(model, optimizer, num_epoch, train_dataset)

    # test the model
    test_fn(model, test_dataset)
