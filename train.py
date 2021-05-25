from tensorflow.keras import preprocessing
import tensorflow as tf
import numpy as np
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import argparse
import os

def get_model(max_id):
    model = tf.keras.models.Sequential([
        tf.keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id], dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id], dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(max_id, activation='softmax'))
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    return parser.parse_known_args()

def get_train_data(train_file):
    with open(os.path.join(train_file, 'macron.txt')) as f:
        text = f.read()
    return text

def preprocess(text, n_steps, batch_size, buffer_size):
    tokenizer = preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(text)
    tokenizer.texts_to_sequences(['First'])
    max_id = len(tokenizer.word_index)
    dataset_size = tokenizer.document_count
    [encoded] = np.array(tokenizer.texts_to_sequences([text])) - 1
    train_size = dataset_size * 90 // 100
    dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
    window_length = n_steps + 1
    dataset = dataset.window(window_length, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_length))
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
    dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
    dataset = dataset.prefetch(1)
    return dataset, max_id

if __name__ == '__main__':
    args, _ = parse_args()
    text = get_train_data(args.train)
    dataset, max_id = preprocess(text, args.n_steps, args.batch_size, args.buffer_size)
    model = get_model(max_id)
    
    model.fit(dataset, epochs=args.epochs)
    model.save(args.model_dir + '/1')
