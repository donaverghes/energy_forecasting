# DNN Model for energy forecasting

import tensorflow as tf
import utils

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
LABEL_COLUMN = 'price'
NUMBER_OF_DAY = 289

# Set default values for each CSV column
DEFAULTS = [['2015-01-01 00:00'], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

def get_feature_columns():
    feature_prediction_date = tf.feature_column.categorical_column_with_hash_bucket(
        CSV_COLUMNS[0], 
        hash_bucket_size=NUMBER_OF_DAY, 
        dtype=tf.string
    )
    
    feature_columns = [ tf.feature_column.indicator_column(feature_prediction_date) ]
                       
    for column in CSV_COLUMNS[1:-2]:
        feature_columns.append(tf.feature_column.numeric_column(column, dtype=tf.float32))
        
        
    return feature_columns

def serving_input_fn():
    feature_placeholders = {
        key: tf.placeholder(tf.float32, [None])
        for key in CSV_COLUMNS
    }
    
    feature_placeholders['prediction_date'] = tf.placeholder(tf.string, [None])
    feature_placeholders.pop('price')
    
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

def my_rmse(labels, predictions):
    pred_values = predictions['predictions']
    return {'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)}

def train_and_evaluate(output_dir, hparams):
    EVAL_INTERVAL = 30
    run_config = tf.estimator.RunConfig(save_checkpoints_secs = EVAL_INTERVAL,
                                        keep_checkpoint_max = 3,
                                       save_summary_steps = 25)
    estimator = tf.estimator.DNNRegressor(
                       model_dir = output_dir,
                       feature_columns = get_feature_columns(),
                       hidden_units = hparams['hidden_units'],
                       config = run_config)
    train_spec = tf.estimator.TrainSpec(
                       input_fn = utils.read_dataset(
                           hparams['train_set'], 
                           mode = tf.estimator.ModeKeys.TRAIN, 
                           batch_size = hparams['batch_size'],
                           defaults = DEFAULTS,
                           csv_columns = CSV_COLUMNS,
                           label_column = LABEL_COLUMN
                       ),
                       max_steps = hparams['training_steps'])
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    estimator = tf.contrib.estimator.add_metrics(estimator, my_rmse)
    eval_spec = tf.estimator.EvalSpec(
                       input_fn = utils.read_dataset(
                           hparams['eval_set'], 
                           mode = tf.estimator.ModeKeys.EVAL,
                           batch_size = hparams['batch_size'],
                           defaults = DEFAULTS,
                           csv_columns = CSV_COLUMNS,
                           label_column = LABEL_COLUMN
                       ),
                       steps = hparams['eval_steps'],
                       start_delay_secs = 60, # start evaluating after N seconds
                       throttle_secs = EVAL_INTERVAL,  # evaluate every N seconds
                       exporters = exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)