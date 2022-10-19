import mindspore as ms
import mindspore.nn as nn
import argparse


from mindspore_.configs import get_config
from mindspore_ import build_model
from mindvision.classification.dataset import Cifar10

def parse_option():
    parser = argparse.ArgumentParser('MSG-Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def main(config):
    download_train = Cifar10(path="./cifar", split="train", batch_size=2, repeat_num=1, shuffle=True, resize=224, download=False)
    dataset_train = download_train.run()

    model = build_model(config)
    # 定义损失函数
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # 定义优化器函数
    net_opt = nn.Momentum(model.trainable_params(), learning_rate=0.01, momentum=0.9)

    network = ms.Model(model, loss_fn=net_loss, optimizer=net_opt)
    network.train(10, dataset_train)


if __name__ == '__main__':
    _, config = parse_option()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")


    main(config)

