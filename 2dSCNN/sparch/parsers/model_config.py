
import logging
from distutils.util import strtobool

logger = logging.getLogger(__name__)


def add_model_options(parser):
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["2dCNN", "2dSCNN"],
        default="2dCNN",
        help="Type of ANN or SNN model.",
    )

    parser.add_argument(
        "--pdrop",
        type=float,
        default=0.1,
        help="Dropout rate, must be between 0 and 1.",
    )

    return parser


def print_model_options(args):
    logging.info(
        """
        Model Config
        ------------
        Model Type: {model_type}
        Dropout rate: {pdrop}
    """.format(
            **vars(args)
        )
    )
