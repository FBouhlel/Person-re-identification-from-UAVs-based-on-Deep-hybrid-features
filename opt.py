import argparse

parser = argparse.ArgumentParser(description='reid')

parser.add_argument('--data_path',
                    default="./PRAI-1581",
                    help='path of PRAI')

parser.add_argument('--mode',
                    default='train', choices=['train', 'evaluate', 'vis'],
                    help='train or evaluate ')

parser.add_argument('--query_image',
                    default='./PRAI-1581/query/00000001_0001_00000001.jpg',
                    help='path to the image you want to query')

parser.add_argument('--freeze',
                    default=False,
                    help='freeze backbone or not ')

parser.add_argument('--weight',
                    default='weights/model.pt',
                    help='load weights ')

parser.add_argument('--epoch',
                    default=100,
                    help='number of epoch to train')

parser.add_argument('--lr',
                    #default=2e-4,
                    default=2e-4,
                    help='initial learning_rate')

parser.add_argument('--lr_scheduler',
                    default=[320, 380],
                    help='MultiStepLR,decay the learning rate')

parser.add_argument("--batchid",
                    default=8,
                    help='the batch for id')

parser.add_argument("--batchimage",
                    default=8,
                    help='the batch of per id')

parser.add_argument("--batchtest",
                    default=1,
                    help='the batch size for test')

opt = parser.parse_args()