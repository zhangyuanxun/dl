import tensorflow as tf
import numpy as np


class LogisticRegression(object):
    def __init__(self):
        self.W = tf.Variable(tf.zeros(shape=[784, 10]), name='W')
        self.b = tf.Variable(tf.zeros(shape=[10]), name='b')

    def __call__(self, X):
        return tf.nn.softmax(tf.matmul(X, self.W) + self.b)


def loss_fn(Y, Y_pred):
    return tf.math.reduce_mean(-tf.math.reduce_sum(Y * tf.math.log(Y_pred)))


def compute_accuracy(Y, Y_pred):
    preds = tf.argmax(Y_pred, axis=1, output_type=tf.int64)
    labels = tf.argmax(Y, axis=1, output_type=tf.int64)
    return tf.reduce_mean(tf.cast(tf.equal(labels, preds), dtype=tf.float32))


def mnist_datasets():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # original size 60000*28*28 -> 60000 * 784
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # normalize dataset
    x_train, x_test = x_train / np.float32(255), x_test / np.float32(255)
    y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)

    # one-hot encoding
    y_train, y_test = np.eye(10)[y_train], np.eye(10)[y_test]
    y_train, y_test = y_train.astype('float32'), y_test.astype('float32')
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_dataset, test_dataset


def train_epoch(model, optimizer, dataset, epoch_loss, epoch_acc):
    for (batch, (inputs, labels)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            preds = model(inputs)
            loss = loss_fn(labels, preds)

        grads = tape.gradient(loss, [model.W, model.b])
        optimizer.apply_gradients(zip(grads, [model.W, model.b]))

        # update metrics
        epoch_loss.update_state(loss)
        epoch_acc.update_state(labels, preds)


def train_fn(model, optimizer, num_epoch, train_dataset):
    display_step = 5
    for epoch in range(num_epoch):
        epoch_loss = tf.keras.metrics.Mean()
        epoch_acc = tf.keras.metrics.CategoricalAccuracy()

        train_epoch(model, optimizer, train_dataset, epoch_loss, epoch_acc)

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

    # load mnist datasets
    train_dataset, test_dataset = mnist_datasets()
    train_dataset = train_dataset.shuffle(num_train).batch(batch_size=batch_size)
    test_dataset = test_dataset.batch(batch_size=batch_size)

    model = LogisticRegression()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    # train the model
    train_fn(model, optimizer, num_epoch, train_dataset)

    # test the model
    test_fn(model, test_dataset)

