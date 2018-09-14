# utils.py: use this file to group commonly used functions
# To use it, import the file: `import utils` and use it as 
# a classic python module: utils.read_dataset(...)
import tensorflow as tf

def read_dataset(filename, mode, batch_size = 512, defaults = None, csv_columns = None, label_column = None):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults=defaults)
            features = dict(zip(csv_columns, columns))
            label = features.pop(label_column)
            return features, label
        # Create list of files that match pattern
        file_list = tf.gfile.Glob(filename)

        # Create dataset from file list
        dataset = (tf.data.TextLineDataset(file_list)  # Read text file
                     .map(decode_csv))  # Transform each elem by applying decode_csv fn

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size=10*batch_size)
        else:
            num_epochs = 1 # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()
    return _input_fn