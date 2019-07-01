import pickle

class ActivationsExtractor():
    def __init__(self, config, model, filename, shard_size=0):
        self.activation_extractors = []
        self.filename = filename
        self.shard_idx = 0
        self.shard_size = shard_size
        self.parse_children(config, model)

    def parse_children(self, config, model, path=""):
        keys = list(config.keys())
        assert len(keys) == 1, "Config file is corrupted"
        name = keys[0]

        path = "%s/%s" % (path, name)

        if config[name]['extract']:
            print("Initializing " + path)
            self.activation_extractors.append(
                SingleModuleActivationsExtractor(path, model, self.flush_shard,
                                           shard_size=self.shard_size,
                                           batch_dim=config[name]['batch_dim'])
            )

        model_child_modules = list(model.named_children())
        assert len(config[name]['children']) == len(model_child_modules), "Config file is corrupted"
        for child_config, (_, child_module) in zip(config[name]['children'], model_child_modules):
            self.parse_children(child_config, child_module, path)

    def flush_shard(self):
        self.save_activations()
        self.shard_idx += 1
        for extractor in self.activation_extractors:
            extractor.flush_activations()

    def save_activations(self):
        filename = self.filename
        if self.shard_size != 0:
            filename = "%s.%d" % (self.filename, self.shard_idx)
        print("Saving activations to %s" % (filename))

        dump = [(extractor.name, extractor.activations) for extractor in self.activation_extractors]
        with open(filename, 'wb') as acts_dump:
            pickle.dump(dump, acts_dump)

## Assumption: Batch size is set to 1
class SingleModuleActivationsExtractor():
    def __init__(self, name, module, global_flush_activations, shard_size=0, batch_dim=0):
        self.activations = []
        self.name = name
        self.batch_dim = batch_dim
        self.hook = module.register_forward_hook(self.capture_activations)
        self.global_flush_activations = global_flush_activations
        self.shard_size = shard_size

    def capture_activations(self, module, inputs, outputs, debug=False):
        if debug:
            print("Capturing activations for:")
            print(module)
            print(inputs.shape + " -> " + outputs.shape)

        if self.shard_size != 0 and len(self.activations)+1 > self.shard_size:
            print("Flushing shard to disk")
            self.global_flush_activations()

        current_acts = outputs.cpu().detach().numpy()
        assert current_acts.shape[self.batch_dim] == 1, \
            ("Invalid batch_dim for module", module)
        current_acts = current_acts.squeeze(axis=self.batch_dim)
        self.activations.append(current_acts)

    def flush_activations(self):
        self.activations = []

    def save_activations(self):
        print("Saving %s activations..." % (self.name))
        with open("%s-activations.pkl" % (self.name), 'wb') as fp:
            pickle.dump(self.activations, fp)

    def remove_hook(self):
        self.hook.remove()

