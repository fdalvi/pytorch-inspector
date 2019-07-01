import json

def extract_structure(model, name="full_model"):
    struct = {
        name: {
            'extract': False,
            'batch_dim': 0,
            'children': []
        }
    }

    for child, child_module in model.named_children():
        struct[name]['children'].append(extract_structure(child_module, name=child))

    return struct

def save_model_config(filename, model, name="full_model"):
    struct = extract_structure(model, name)
    with open(filename, 'w') as config_file:
        json.dump(struct, config_file, indent=2)

def load_model_config(filename):
    with open(filename) as config_file:
        struct = json.load(config_file)
        return struct