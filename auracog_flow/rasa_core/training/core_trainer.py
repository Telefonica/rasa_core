import argparse
import logging

from auracog_flow.series_dialogue.policy import SeriesPolicy
from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.policies.fallback import FallbackPolicy

logger = logging.getLogger(__name__)


# TODO: add support for parameter "unique_last_num_states"
# TODO: add support for parameter "use_story_concatenation",
# TODO: add support for interactive training

# Core dialogues default threshold
CORE_THRESHOLD = 0.5
# NLU default threshold
NLU_THRESHOLD = 0.5

def create_argument_parser():
    """Parse all the command line arguments for the training script."""

    parser = argparse.ArgumentParser(
            description='trains a dialogue model')
    parser.add_argument(
            '-s', '--stories',
            type=str,
            required=True,
            help="file or folder containing the training stories")
    parser.add_argument(
            '-o', '--out',
            type=str,
            required=True,
            help="directory to persist the trained model in")
    parser.add_argument(
            '-d', '--domain',
            type=str,
            required=True,
            help="domain specification yaml file")
    parser.add_argument(
            '--max_history',
            type=int,
            default=3,
            help="max history to use of a story")
    parser.add_argument(
            '--epochs',
            type=int,
            default=100,
            help="number of epochs to train the model")
    parser.add_argument(
            '--validation_split',
            type=float,
            default=0.2,
            help="Percentage of training samples used for validation, "
                 "0.2 by default")
    parser.add_argument(
            '--batch_size',
            type=int,
            default=20,
            help="number of training samples to put into one training batch")
    parser.add_argument(
            '--augmentation',
            type=int,
            default=50,
            help="how much data augmentation to use during training")
    parser.add_argument(
            '--debug_plots',
            default=False,
            action='store_true',
            help="If enabled, will create plots showing checkpoints "
                 "and their connections between story blocks in a  "
                 "file called `story_blocks_connections.pdf`.")
    parser.add_argument(
            '--tracker_limit',
            type=int,
            default=None,
            help="This option is not yet documented in Rasa source code. It seems to not be used either.")
    parser.add_argument(
            '--exclusion_percentage',
            type=int,
            default=None,
            help="Percentage of stories to be left out during data loading. If not pecified, 100% is used.")
    parser.add_argument(
            '--rnn_size',
            type=int,
            default=32,
            help="Number of LSTM units in hidden layer.")
    parser.add_argument(
            "--use_story_concatenation",
            default=True,
            action="store_true",
            help="Use story concatenation."
    )
    parser.add_argument(
            "--core_threshold",
            type=float,
            default=CORE_THRESHOLD,
            help="Fallback core dialogues threshold"
    )
    parser.add_argument(
            "--nlu_threshold",
            type=float,
            default=NLU_THRESHOLD,
            help="Fallback NLU threshold"
    )

    tracker_limit = None,  # type: Optional[int]
    exclusion_percentage = None  # type: int


    utils.add_logging_option_arguments(parser)
    return parser


def train_dialogue_model(domain_file, stories_file, output_path,
                         max_history=None,
                         batch_size=100,
                         epochs=200,
                         validation_split=0.2,
                         rnn_size=32,
                         core_threshold=CORE_THRESHOLD,
                         nlu_threshold=NLU_THRESHOLD,
                         kwargs=None):
    """

    :param domain_file:
    :param stories_file:
    :param output_path:
    :param nlu_model_path:
    :param max_history:
    :param batch_size:
    :param epochs:
    :param validation_split:
    :param kwargs:
    :return:
    """
    if not kwargs:
        kwargs = {}

#    agent = Agent(domain_file,
#                  policies=[MemoizationPolicy(max_history=max_history),
#                            SeriesPolicy(batch_size=batch_size, epochs=epochs,
#                                         validation_split=validation_split,
#                                         rnn_size=rnn_size)])

    fallback = FallbackPolicy(fallback_action_name="action_fallback",
                          core_threshold=core_threshold,
                          nlu_threshold=nlu_threshold)
    agent = Agent(domain_file,
                  policies=[MemoizationPolicy(max_history=max_history),
                            SeriesPolicy(batch_size=batch_size, epochs=epochs,
                                         validation_split=validation_split,
                                         rnn_size=rnn_size),
                        fallback
                        ])

    # data_load_args, kwargs = utils.extract_args(kwargs,
    #                                             {"use_story_concatenation",
    #                                              "unique_last_num_states",
    #                                              "augmentation_factor",
    #                                              "remove_duplicates",
    #                                              "debug_plots"})
    data_load_args, kwargs = utils.extract_args(kwargs,
                                                {"use_story_concatenation",
                                                 "unique_last_num_states",
                                                 "augmentation_factor",
                                                 "remove_duplicates",
                                                 "debug_plots",
                                                 "tracker_limit",
                                                 "exclusion_percentage"})

    _, _kwargs = utils.extract_args(kwargs, {"batch_size", "epochs", "validation_split"})

    training_data = agent.load_data(stories_file, **data_load_args)
#    agent.train(training_data, **kwargs)
    agent.train(training_data, **_kwargs)
    agent.persist(output_path)


if __name__ == '__main__':

    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    utils.configure_colored_logging(cmdline_args.loglevel)

    additional_arguments = {
        "epochs": cmdline_args.epochs,
        "batch_size": cmdline_args.batch_size,
        "validation_split": cmdline_args.validation_split,
        "augmentation_factor": cmdline_args.augmentation,
        "debug_plots": cmdline_args.debug_plots,
        "tracker_limit": cmdline_args.tracker_limit,
        "exclusion_percentage": cmdline_args.exclusion_percentage
    }

    print("Start training with the following parameters: {}".format(cmdline_args))

    train_dialogue_model(cmdline_args.domain,
                         cmdline_args.stories,
                         cmdline_args.out,
                         cmdline_args.max_history,
                         cmdline_args.batch_size,
                         cmdline_args.epochs,
                         cmdline_args.validation_split,
                         cmdline_args.rnn_size,
                         cmdline_args.core_threshold,
                         cmdline_args.nlu_threshold,
                         additional_arguments)
