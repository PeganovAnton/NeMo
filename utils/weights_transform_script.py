from collections import OrderedDict
from sys import argv
import torch
from omegaconf import OmegaConf
from nemo.collections.nlp.models import MTEncDecModel


def main() -> None:
    config_yaml = argv[1]
    model_weights = argv[2]
    new_nemo_file = argv[3]
    conf = OmegaConf.load(config_yaml)
    mt_model = MTEncDecModel.from_config_dict(config=conf)
    weights_dict = torch.load(model_weights)
    new_dict = dict()
    for key, value in weights_dict.items():
        if key.startswith('encoder.'):
            new_dict[key.replace('encoder.', 'encoder._encoder.')] = value
        elif key.startswith('decoder.'):
            new_dict[key.replace('decoder.', 'decoder._decoder.')] = value
        elif key.startswith('encoder_embedding'):
            new_dict[key.replace('encoder_embedding', 'encoder._embedding')] = value
        elif key.startswith('decoder_embedding'):
            new_dict[key.replace('decoder_embedding', 'decoder._embedding')] = value
        else:
            new_dict[key] = value
    mt_model.load_state_dict(OrderedDict(new_dict), strict=True)
    mt_model.save_to(save_path=new_nemo_file)


if __name__ == '__main__':
    main()