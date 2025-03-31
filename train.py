import os
os.system('clear')
print()

import argparse
import yaml
import torch
from tqdm import trange

from model.models.detection_model import DetectionModel
from model.data.dataset import Dataset

from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='YOLOv8 model training')
    parser.add_argument(
        '--model-config',
        type=str,
        default='/home/ajinkya.kulkarni/yolov8/model/config/models/celldetection.yaml',
        help='path to model config file'
    )
    parser.add_argument(
        '--weights',
        type=str,
        help='path to weights file'
    )

    parser.add_argument(
        '--train-config',
        type=str,
        default='/home/ajinkya.kulkarni/yolov8/model/config/models/celldetection.yaml',
        help='path to training config file'
    )

    dataset_args = parser.add_argument_group('Dataset')
    dataset_args.add_argument(
        '--dataset',
        type=str,
        default='/home/ajinkya.kulkarni/yolov8/celldetectiondataset/dataset.yaml',
        help='path to dataset config file'
    )
    dataset_args.add_argument(
        '--dataset-mode',
        type=str,
        default='train',
        help='dataset mode'
    )

    parser.add_argument(
        '--device',
        '-d',
        type=str,
        default='cuda',
        help='device to model on'
    )

    return parser.parse_args()


def main(args):
    train_config = yaml.safe_load(open(args.train_config, 'r'))

    device = torch.device(args.device)
    model = DetectionModel(args.model_config, device=device)
    if args.weights is not None:
        model.load(torch.load(args.weights))

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])

    dataset = Dataset(args.dataset, mode=args.dataset_mode, batch_size=train_config['batch_size'])
    dataloader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=True, collate_fn=Dataset.collate_fn)

    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), train_config['save_dir'], os.path.splitext(os.path.basename(args.model_config))[0])
    os.makedirs(save_path, exist_ok=True)

    print()

    for epoch in trange(train_config['epochs'], desc='Training', leave=True):
        for batch in dataloader:
            loss = model.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % train_config['save_freq'] == 0:
            model.save(os.path.join(save_path, f'{epoch+1}.pt'))


if __name__ == '__main__':
    args = get_args()
    main(args)
    