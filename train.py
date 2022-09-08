import argparse
import torch
import torch.backends.cudnn as cudnn
import os
from trainer import SynGAN_Trainer
import yaml
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# python -m visdom.server -p 6019


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default=r'G:SynGAN\Yaml\P2p.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)

    config['save_root'] = os.path.join(config['save_root'], datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    config['save_checkpoint'] = os.path.join(config['save_root'], 'checkpoint')
    if not os.path.exists(config['save_checkpoint']):
        os.makedirs(config['save_checkpoint'])
    
    trainer = SynGAN_Trainer(config)
    trainer.train_summary()


###################################
if __name__ == '__main__':
    main()
