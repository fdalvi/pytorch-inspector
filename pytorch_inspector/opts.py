import sys

from enum import Enum

class Mode(Enum):
    extract_structure = 'extract_structure'
    extract_activations = 'extract_activations'

    def __str__(self):
        return self.value

def add_opts(parser):
    parser.add_argument('--mode', type=Mode, choices=list(Mode), required=True,
                        help="""Set to 'extract_structure' to save the model structure JSON
                                or 'extract_activations' to extract activations based on a
                                saved model structure JSON""")
    parser.add_argument('--config_file', type=str, required=True,
                        help="""Path to JSON file (will either be saved if mode is
                                'extract_structure' or loaded is mode is 'extract_activations'""")
    parser.add_argument('--output_activations', type=str,
                        help="""Path where output activations must be saved""")
    parser.add_argument('--activations_shard_size', type=int, default=0,
                        help="""Divide extracted activations into shards. This can help with
                             memory issues, since each shard will be flushed to disk before
                             computing the next shard. Set to 0 to disable sharding (default)""")

def check_opts(opts):
    if opts.mode == Mode.extract_activations and opts.output_activations is None:
        print("""Please specify a path to save the output activations to
                 (using --output_activations)""")
        sys.exit(1)

    if opts.mode == Mode.extract_activations and opts.output_activations[-5:] != ".acts":
        opts.output_activations = "%s.acts" % (opts.output_activations)

    if opts.batch_size != 1:
        print("Warning: Extractor only works with batch_size = 1. Ignoring batch_size parameter.")
        opts.batch_size = 1
