# CNN Model for energy forecasting

import tensorflow as tf
import utils
from logging import INFO

tf.logging.set_verbosity(INFO)

CSV_COLUMNS = [ 'prediction_date',
               'avg_wind_speed_100m',
               'avg_wind_direction_100m',
               'avg_temperature',
               'avg_air_density',
               'avg_pressure',
               'avg_precipitation',
               'avg_wind_gust',
               'avg_radiation',
               'avg_wind_speed',
               'avg_wind_direction',
               'price',
               'key'
              ]
# Set default values for each CSV column
DEFAULTS = [['2015-01-01 00:00'], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
TIMESERIES_COL = 'price'
LABEL_COLUMN = 'price'
N_OUTPUTS = 1  # Predict the price for the next hour
SEQ_LEN = 24
N_INPUTS = SEQ_LEN - N_OUTPUTS

def init(hparams):
    global SEQ_LEN, DEFAULTS, N_INPUTS
    SEQ_LEN = hparams['sequence_length']
    N_INPUTS = SEQ_LEN - N_OUTPUTS
    
def cnn_model(features, mode, params):
    X = tf.reshape(features[TIMESERIES_COL],
                   [-1, N_INPUTS, 1])  # as a 1D "sequence" with only one time-series observation (height)
    c1 = tf.layers.conv1d(X, filters=N_INPUTS // 2,
                          kernel_size=3, strides=1,
                          padding='same', activation=tf.nn.relu)
    p1 = tf.layers.max_pooling1d(c1, pool_size=2, strides=2)

    c2 = tf.layers.conv1d(p1, filters=N_INPUTS // 2,
                          kernel_size=3, strides=1,
                          padding='same', activation=tf.nn.relu)
    p2 = tf.layers.max_pooling1d(c2, pool_size=2, strides=2)

    outlen = p2.shape[1] * p2.shape[2]
    c2flat = tf.reshape(p2, [-1, outlen])
    h1 = tf.layers.dense(c2flat, 3, activation=tf.nn.relu)
    predictions = tf.layers.dense(h1, 1, activation=None)  # linear output: regression
    return predictions
#    X = tf.reshape(features[TIMESERIES_COL], [-1, (SEQ_LEN - N_OUTPUTS), 1])
#    
#    conv_1 = tf.layers.conv1d(X, 
#                              filters=(SEQ_LEN - N_OUTPUTS) // 2, 
#                              kernel_size=params['kernel_size'], 
#                              strides=1,
#                              padding='same',
#                              activation=tf.nn.relu
#                             )
#    max_pool_1 = tf.layers.max_pooling1d(conv_1,
#                                         pool_size=2,
#                                         strides=2,
#                                        )
#    conv_2 = tf.layers.conv1d(max_pool_1,
#                              filters=(SEQ_LEN - N_OUTPUTS) // 2,
#                              kernel_size=params['kernel_size'],
#                              strides=1,
#                              padding='same',
#                              activation=tf.nn.relu
#                             )
#    max_pool_2 = tf.layers.max_pooling1d(conv_2,
#                                         pool_size=2,
#                                         strides=2,
#                                        )
#    
#    outlen = max_pool_2.shape[1] * max_pool_2.shape[2]
#    vflat = tf.reshape(max_pool_2, [-1, outlen])
#    nn = tf.layers.dense(max_pool_2, units=3, activation=tf.nn.relu)
#    outputs = tf.layers.dense(nn, 1, activation=None)
#    return outputs

def serving_input_fn():
    feature_placeholders = {
        TIMESERIES_COL: tf.placeholder(tf.float32, [None, N_INPUTS])
    }

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    features[TIMESERIES_COL] = tf.squeeze(features[TIMESERIES_COL], axis=[2])

    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

def compute_errors(features, labels, predictions):
    labels = tf.expand_dims(labels, -1)
    
    tf.logging.info("Shape of label: {}, Shape of predictions: {}".format(labels.shape, predictions.shape))
    if predictions.shape[1] == 1:
        loss = tf.losses.mean_squared_error(labels, predictions)
        rmse = tf.metrics.root_mean_squared_error(labels, predictions)
        return loss, rmse
    else:
        labelsN = tf.concat([features[TIMESERIES_COL], labels], axis=1)
        labelsN = labelsN[:, 1:]
        N = (2 * (SEQ_LEN - N_OUTPUTS)) // 3
        tf.logging.info("Shape of labelsN: {}, Shape of predictions: {}".format(labelsN[:, N:].shape, predictions[:, N:].shape))
        loss = tf.losses.mean_squared_error(labelsN[:, N:], predictions[:, N:])
        last_pred = predictions[:, -1]
        rmse = tf.metrics.root_mean_squared_error(labels, last_pred)
        return loss, rmse
    
def same_as_last_benchmark(features, labels):
    predictions = features[TIMESERIES_COL][:,-1] # last value in input sequence
    return tf.metrics.root_mean_squared_error(labels, predictions)

# create the inference model
def sequence_regressor(features, labels, mode, params):
    predictions = cnn_model(features, mode, params)

    loss = None
    train_op = None
    eval_metric_ops = None
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        loss, rmse = compute_errors(features, labels, predictions)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # this is needed for batch normalization, but has no effect otherwise
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # 2b. set up training operation
                train_op = tf.contrib.layers.optimize_loss(
                    loss,
                    tf.train.get_global_step(),
                    learning_rate=params['learning_rate'],
                    optimizer="Adam")

        # 2c. eval metric
        eval_metric_ops = {
            "rmse": rmse,
            "rmse_same_as_last": same_as_last_benchmark(features, labels),
        }

    # 3. Create predictions
    if predictions.shape[1] != 1:
        predictions = predictions[:, -1]  # last predicted value
    predictions_dict = {"predicted": predictions}

    # 4. return EstimatorSpec
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs={
            'predictions': tf.estimator.export.PredictOutput(predictions_dict)}
    )

def train_and_evaluate(output_dir, hparams):
    EVAL_INTERVAL = 30
    run_config = tf.estimator.RunConfig(save_checkpoints_secs = EVAL_INTERVAL,
                                        keep_checkpoint_max = 3,
                                        save_summary_steps = 25)
    estimator = tf.estimator.Estimator(
                       model_fn = sequence_regressor,
                       params = hparams,
                       model_dir = output_dir,
                       config = run_config)
    train_spec = tf.estimator.TrainSpec(
                       input_fn = utils.read_dataset(
                           hparams['train_set'], 
                           mode = tf.estimator.ModeKeys.TRAIN, 
                           batch_size = hparams['batch_size'],
                           timeserie_column = TIMESERIES_COL
                       ),
                       max_steps = hparams['training_steps'])
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
                       input_fn = utils.read_dataset(
                           hparams['eval_set'], 
                           mode = tf.estimator.ModeKeys.EVAL,
                           batch_size = hparams['batch_size'],
                           timeserie_column = TIMESERIES_COL
                       ),
                       steps = hparams['eval_steps'],
                       start_delay_secs = 60, # start evaluating after N seconds
                       throttle_secs = EVAL_INTERVAL,  # evaluate every N seconds
                       exporters = exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)