import argparse
import logging

from rasa_nlu.training_data import load_data
from rasa_nlu import config
from rasa_nlu.model import Trainer
from rasa_nlu import utils

"""
training_data = load_data('data/nlu_data.md')
trainer = Trainer(config.load("nlu_model_config.yml"))
trainer.train(training_data)
model_directory = trainer.persist('models/nlu/',
                                  fixed_model_name="current")

return model_directory
"""

def create_argument_parser():
    """Parse all the command line arguments for the training script."""

    parser = argparse.ArgumentParser(
            description="Trains a Rasa NLU model")
    parser.add_argument(
            '-c', '--config_file',
            type=str,
            required=True,
            help="File containing the model configuration")
    parser.add_argument(
            '-d', '--data_file',
            type=str,
            required=True,
            help="File containing the training data.")
    parser.add_argument(
            '-o', '--out',
            type=str,
            required=True,
            help="Directory to persist the trained model in")
    parser.add_argument(
            '-p', '--project_name',
            type=str,
            required=True,
            help="Project name.")
    parser.add_argument(
            '-m', '--model_name',
            type=str,
            required=True,
            help="Model name.")
    parser.add_argument(
            '--epochs',
            type=int,
            default=100,
            help="number of epochs to train the model")
    parser.add_argument(
            '--validation_split',
            type=float,
            default=0.1,
            help="Percentage of training samples used for validation, "
                 "0.1 by default")
    parser.add_argument(
            '--batch_size',
            type=int,
            default=20,
            help="number of training samples to put into one training batch")

    utils.add_logging_option_arguments(parser)
    return parser


def train_nlu_model(config_file, data_file, output_path, project_name, model_name, kwargs=None):
    """
    Trains a NLU model.

    :param config_file: YAML file containing the model configuration.
    :param data_file: The file containing the training data (md or JSON format).
    :param output_path: Directory to persist the trained model.
    :param model_name: Model name.
    :param kwargs:
    :return:
    """
    if not kwargs:
        kwargs = {}

    training_data = load_data(data_file)
    trainer = Trainer(config.load(config_file))
    trainer.train(training_data, **kwargs)
    model_directory = trainer.persist(output_path, project_name=project_name,
                                      fixed_model_name=model_name)
    return model_directory


if __name__ == '__main__':

    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    utils.configure_colored_logging(cmdline_args.loglevel)

    additional_arguments = {
        "epochs": cmdline_args.epochs,
        "validation_split": cmdline_args.validation_split,
        "batch_size": cmdline_args.batch_size
    }

    train_nlu_model(cmdline_args.config_file,
                    cmdline_args.data_file,
                    cmdline_args.out,
                    cmdline_args.project_name,
                    cmdline_args.model_name,
                    additional_arguments)
