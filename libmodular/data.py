import reader
import tensorflow as tf


class DataLoader:

    def __init__(self, data_dir, batch_size):
        self._load_dataset(data_dir, batch_size)

    def _load_dataset(self, data_dir, batch_size):
        train_data, valid_data, test_data, self.vocabulary = reader.ptb_raw_data(str(data_dir))
        self.train_dataset, self.train_epoch_size = reader.ptb_producer(train_data, batch_size, num_steps=35)
        self.test_dataset, self.test_epoch_size = reader.ptb_producer(test_data, batch_size, num_steps=35,
                                                                      repeat=False)
        self.validation_dataset, self.validation_epoch_size = reader.ptb_producer(valid_data, batch_size,
                                                                                  num_steps=35, repeat=False)

        self.handle = tf.placeholder(tf.string, shape=[], name='dataset_handle')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        iterator = tf.data.Iterator.from_string_handle(self.handle, self.train_dataset.output_types,
                                                       self.train_dataset.output_shapes)
        self.inp, self.target = iterator.get_next()
