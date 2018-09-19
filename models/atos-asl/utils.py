# utils.py: use this file to group commonly used functions
# To use it, import the file: `import utils` and use it as 
# a classic python module: utils.read_dataset(...)

import tensorflow as tf

# Set default values for each CSV column
DEFAULTS_CSV = [['2015-01-01 00:00'], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
SEQ_LEN = 24 # Weekly timeserie
DEFAULTS_TS = [ [0.0] for i in range(SEQ_LEN)]

def read_dataset(filename, mode, batch_size = 512, defaults = None, csv_columns = None, label_column = None, timeserie_column=None):
    def _input_fn():
        def decode_csv(value_column):
            #tf.logging.info('defaults: {}, csv_columns: {}, label_column: {}, timeserie_column: {}'.format(defaults, csv_columns, label_column, timeserie_column))
            if timeserie_column is not None: # Timeserie mode, read the row and the last one is the label
                features = tf.decode_csv(value_column, record_defaults=DEFAULTS_TS)  # string tensor -> list of 50 rank 0 float tensors
                label = features.pop()  # remove last feature and use as label
                features = tf.stack(features)
                return {timeserie_column: features}, label
            else:
                columns = tf.decode_csv(value_column, record_defaults=DEFAULTS_CSV)
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