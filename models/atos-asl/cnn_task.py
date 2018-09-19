# task.py: main interface to the models
import argparse
import json
import os
import cnn_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--eval_set',
        help = 'GCS path to eval dataset.',
        required = True
    )
    parser.add_argument(
    '--train_set',
    help = 'GCS path to train dataset.',
    required = True
    )
    parser.add_argument(
        '--output_dir',
        help = 'GCS location to write checkpoints and export models',
        required = True
    )
    parser.add_argument(
        '--batch_size',
        help = 'Number of examples to compute gradient over.',
        type = int,
        default = 512
    )
    parser.add_argument(
        '--sequence_length',
        help = 'Length of the sequence to use to get prediction',
        type = int,
        default = 24
    )
    parser.add_argument(
        '--job-dir',
        help = 'this model ignores this field, but it is required by gcloud',
        default = 'junk'
    )
    parser.add_argument(
        '--nfilters',
        help = 'Number of filters to use in the CNN layers',
        type = int,
        default=6
    )
    parser.add_argument(
        '--kernel_size',
        help = 'Size of the kernel to use in the CNN layers',
        type = int,
        default = 3
    )
    parser.add_argument(
        '--training_steps',
        help = 'Number of training steps',
        type = int,
        default = 10000
    )
    parser.add_argument(
        '--learning_rate',
        help = 'The learning rate',
        type = float,
        default = 0.1
    )
    parser.add_argument(
        '--eval_steps',
        help = 'Positive number of steps for which to evaluate model. Default to None, which means to evaluate until input_fn raises an end-of-input exception',
        type = int,       
        default = None
    )
        
    ## parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__

    # unused args provided by service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)

    ## assign the arguments to the model variables
    output_dir = arguments.pop('output_dir')

    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    output_dir = os.path.join(
        output_dir,
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trial', '')
    )

    # Run the training job
    cnn_model.train_and_evaluate(output_dir, arguments)