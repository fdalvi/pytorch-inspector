pytorch-inspector
=================

A generic package to inspect and extract activations from deep learning models built in pytorch. The code allows you to:
1: Extract the architecture of the model in a human-readable JSON file
2: Extract activations from some or all of the intermediate modules (layers) of the model

### Installation and Running
_`pip` support coming soon_
- Copy the `pytorch_inspector` directory into the root of your code
- Import `add_opts` and `check_opts` from `pytorch_inspector.opts`
    - `add_opts` takes an `ArgumentParser` (`argparse` lib) and adds the options necessary for the inspector. Add this when you are defining your `ArgumentParser` but before your parse the actual input (before `parse_args()` call)
    - `check_opts` checks if the conditions for the inspector are met - Add this after your `parse_args()` call
- Import `Mode` from `pytorch_inspector.opts` and `load_model_config`/`save_model_config` from `pytorch_inspector.structure`
    - You can use `opt.mode` to see if the user requested for model structure extraction (`= Mode.extract_structure`) or activations extraction (`= Mode.extract_activations`)
    - if the mode is model structure extraction, you can call `save_model_config` to save the model architecture in a JSON file
    - if the mode is model activation extraction, you can call `load_model_config` to load the model configuration, and initialize an `ActivationsExtractor` (`pytorch_inspector.extractor.ActivationsExtractor`)
    - After your forward pass and before you exit the code, perform a final call to `ActivationsExtractor.save_activations()` to save the activations

### Sample clients
We have already implemented the extractor for some existing code bases:
- OpenNMT-py: `clients/OpenNMT-py/translate.py`

### Miscellaneous Notes
- Support for shards comes out of the box. The extractor saves activations in memory before dumping them to disk - if your inputset is large and you cannot fit all the activations in memory, you can use sharding to periodically dump everything to disk after `N` instances
- Currently, the extractor only supports `batch_sizes` = 1, but support for larger batches is coming soon

### API
Coming soon
