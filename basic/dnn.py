import tensorflow as tf
import numpy as np


class DNN(object):
    # 2 hidden layer deep neural network
    def __init__(self):
        self.num_hidden1 = 256
        self.num_hidden2 = 128
        self.num_inputs = 784
        self.num_class = 10

        self.weights = {
            'W1': tf.Variable(tf.random.normal(shape=[self.num_inputs, self.num_hidden1], stddev=0.1)),
            'W2': tf.Variable(tf.random.normal(shape=[self.num_hidden1, self.num_hidden2], stddev=0.1)),
            'out': tf.Variable(tf.random.normal(shape=[self.num_hidden2, self.num_class], stddev=0.1)),
        }

        self.biases = {
            'b1': tf.Variable(tf.random.normal(shape=[self.num_hidden1])),
            'b2': tf.Variable(tf.random.normal(shape=[self.num_hidden2])),
            'out': tf.Variable(tf.random.normal(shape=[self.num_class])),
        }

    def __call__(self, X):
        layer1 = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['W1']), self.biases['b1']))
        layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, self.weights['W2']), self.biases['b2']))
        return tf.add(tf.matmul(layer2, self.weights['out']), self.biases['out'])

    def trainable_variables(self):
        return [self.weights['W1'], self.weights['W2'], self.weights['out'],
                self.biases['b1'], self.biases['b2'], self.biases['out']]


def loss_fn(Y, Y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, Y_pred))



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


def train_step(model, optimizer, dataset, epoch_loss, epoch_acc):
    for (batch, (inputs, labels)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            preds = model(inputs)
            loss = loss_fn(labels, preds)

        grads = tape.gradient(loss, model.trainable_variables())
        optimizer.apply_gradients(zip(grads, model.trainable_variables()))

        # update metrics
        epoch_loss.update_state(loss)
        epoch_acc.update_state(labels, preds)


def train_fn(model, optimizer, num_epoch, train_dataset):
    display_step = 5
    for epoch in range(num_epoch):
        epoch_loss = tf.keras.metrics.Mean()
        epoch_acc = tf.keras.metrics.CategoricalAccuracy()

        train_step(model, optimizer, train_dataset, epoch_loss, epoch_acc)

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
    num_epoch = 50
    batch_size = 100
    num_train = 60000
    num_test = 10000

    # load mnist datasets
    train_dataset, test_dataset = mnist_datasets()
    train_dataset = train_dataset.shuffle(num_train).batch(batch_size=batch_size)
    test_dataset = test_dataset.batch(batch_size=batch_size)

    model = DNN()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.005)

    # train the model
    train_fn(model, optimizer, num_epoch, train_dataset)

    # test the model
    test_fn(model, test_dataset)



