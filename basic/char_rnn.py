import tensorflow as tf
import numpy as np
import time
import math


def load_dataset():
    print('Load Shakespeare Dataset...')
    dataset_path = tf.keras.utils.get_file('shakespeare.txt',
                                           'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    text = open(dataset_path, 'rb').read().decode(encoding='utf-8')
    print('Length of text: {} characters'.format(len(text)))

    # Take a look at the first 250 characters in text
    print(text[:250])

    return text


def process_dataset(text):
    vocab = sorted(set(text))

    # Creating a mapping from unique characters to indices
    char2id = {u: i for i, u in enumerate(vocab)}
    id2char = np.array(vocab)
    text2int = np.array([char2id[c] for c in text])

    # Show how the first 13 characters from the text are mapped to integers
    print('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text2int[:13]))

    return text2int, char2id, id2char, len(vocab)


def create_train_dataset(text2int, id2char, batch_size):
    seq_length = 35
    BUFFER_SIZE = 100
    char_dataset = tf.data.Dataset.from_tensor_slices(text2int)

    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]  # one character shift

        return input_text, target_text

    train_dataset = sequences.map(split_input_target)

    for input_example, target_example in train_dataset.take(1):
        print('Input data: ', repr(''.join(id2char[input_example.numpy()])))
        print('Target data:', repr(''.join(id2char[target_example.numpy()])))

    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)

    return train_dataset


class RNN(object):
    def __init__(self, vocab_size, num_hidden):
        self.vocab_size = vocab_size
        self.num_hidden = num_hidden

        # initialize weights for hidden
        self.W_xh = tf.Variable(tf.random.normal(shape=(vocab_size, num_hidden), stddev=0.01, mean=0, dtype=tf.float32))
        self.W_hh = tf.Variable(tf.random.normal(shape=(num_hidden, num_hidden), stddev=0.01, mean=0, dtype=tf.float32))
        self.b_h = tf.Variable(tf.zeros(num_hidden), dtype=tf.float32)

        # initialize weights for output
        self.W_hy = tf.Variable(tf.random.normal(shape=(num_hidden, vocab_size), stddev=0.01, mean=0, dtype=tf.float32))
        self.b_y = tf.Variable(tf.zeros(vocab_size), dtype=tf.float32)

    def __call__(self, X, state):
        H = state
        Y = []
        for x in X:
            x = tf.reshape(x, [-1, self.W_xh.shape[0]])
            H = tf.math.tanh(tf.matmul(x, self.W_xh) + tf.matmul(H, self.W_hh) + self.b_h)
            y = tf.matmul(H, self.W_hy) + self.b_y
            Y.append(y)

        return Y, H

    def trainable_variables(self):
        return [self.W_xh, self.W_hh, self.W_hy, self.b_h, self.b_y]

    def init_state(self, size):
        return tf.zeros(shape=(size, num_hidden))


def to_onehot(X, size):
    return tf.one_hot(tf.transpose(X), size, dtype=tf.float32)


def grad_clipping(grads, theta):
    norm = np.array([0])
    for g in grads:
        norm += tf.math.reduce_sum(g**2)

    norm = np.sqrt(norm).item()

    new_grads = []
    if norm > theta:
        for g in grads:
            new_grads.append(g * theta / norm)
        return new_grads
    else:
        return grads


def train_fn(train_dataset, model, optimizer, num_epoch, vocab_size,
             clipping_theta, display_step, batch_size):

    for epoch in range(num_epoch):
        total_loss, num_sample, start = 0, 0, time.time()
        state = model.init_state(size=batch_size)
        for batch, (X, Y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                inputs = to_onehot(X, vocab_size)

                # run rnn model, outputs is list with (batch_size, vocab_size)
                outputs, state = model(inputs, state)

                # after concatenating, the size is (num_sequence * batch_size, vocab_size)
                outputs = tf.concat(outputs, 0)

                # reshape target Y
                Y = tf.reshape(tf.transpose(Y), [-1, ])

                # compute loss
                loss = tf.math.reduce_mean(tf.losses.sparse_categorical_crossentropy(Y, outputs))

            grads = tape.gradient(loss, model.trainable_variables())
            grads = grad_clipping(grads, clipping_theta)
            optimizer.apply_gradients(zip(grads, model.trainable_variables()))
            total_loss += loss.numpy().item() * len(Y)
            num_sample += len(Y)

        if (epoch + 1) % display_step == 0:
            print('Epoch %d: perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(total_loss / num_sample), time.time() - start))


def generate_fn(model, char2id, id2char, vocab_size, start_text):
    # Number of characters to generate
    num_generate = 200

    # use last output as current input
    inputs = [char2id[c] for c in start_text]
    inputs = tf.convert_to_tensor(to_onehot(inputs, vocab_size))

    text_generated = []

    state = model.init_state(size=1)

    for t in range(num_generate):
        outputs, state = model(inputs, state)

        predict_id = tf.random.categorical(outputs[-1], num_samples=1)[-1, 0].numpy()
        #predict_id = int(np.array(tf.argmax(outputs[-1], axis=1)))
        text_generated.append(id2char[predict_id])
        inputs = outputs[-1]

    return start_text + ''.join(text_generated)


if __name__ == '__main__':
    # Define hyper-parameters
    batch_size = 32
    num_hidden = 512
    num_epoch = 50

    # Load Shakespeare Dataset
    text = load_dataset()

    # Data processing
    text2int, char2id, id2char, vocab_size = process_dataset(text)

    # create training dataset
    train_dataset = create_train_dataset(text2int, id2char, batch_size)

    model = RNN(vocab_size=vocab_size, num_hidden=num_hidden)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.6)

    train_fn(train_dataset=train_dataset, model=model,
             optimizer=optimizer, num_epoch=num_epoch,
             vocab_size=vocab_size, clipping_theta=1e-2,
             display_step=1, batch_size=batch_size)

    generated_text = generate_fn(model=model, char2id=char2id, id2char=id2char,
                                 vocab_size=vocab_size, start_text=u"ROMEO: ")

    print(generated_text)