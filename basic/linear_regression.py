import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(object):
    def __init__(self):
        self.W = tf.Variable(tf.random.uniform(shape=[1], minval=-1, maxval=1), name='W')
        self.b = tf.Variable(tf.zeros(shape=[1]), name='b')

    def __call__(self, x):
        return self.W * x + self.b


def loss_fn(Y, Y_pred):
    return tf.reduce_mean(tf.square(Y - Y_pred))


def train(model, optimizer, epoch, inputs, labels):
    with tf.GradientTape() as tape:
        pred = model(inputs)
        loss = loss_fn(pred, labels)
    gradients = tape.gradient(loss, [model.W, model.b])
    optimizer.apply_gradients(zip(gradients, [model.W, model.b]))
    print('epoch %d\t loss: %.6f' % (epoch, loss))


if __name__ == '__main__':
    # generate random dataset
    num_points = 1000
    points = []
    W, b = 0.1, 0.3
    for i in range(num_points):
        x = np.random.normal(0.0, 0.55)
        y = x * W + b + np.random.normal(0.0, 0.03)
        points.append([x, y])

    # generate x, y dataset
    x_data = [p[0] for p in points]
    y_data = [p[1] for p in points]

    # plot this line
    plt.scatter(x_data, y_data, c='r')
    plt.show()
    num_epoch = 50

    model = LinearRegression()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.3)

    print("Initial Weight:")
    print("W = %.4f, b = %.4f" % (model.W, model.b))
    for i in range(num_epoch):
        train(model, optimizer, i + 1, x_data, y_data)

    print("Final Weight:")
    print("W = %.4f, b = %.4f" % (model.W, model.b))

    # plot the fitted line
    plt.scatter(x_data, y_data, c='r')
    plt.plot(x_data, model.W * x_data + model.b, c='b')
    plt.show()
