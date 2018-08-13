import argparse
import torch
import os

from utils import utils


class OptionParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def parse_args(self):
        opt = self.parser.parse_args()
        str_ids = str(opt.gpu_ids)
        opt.gpu_ids = []
        for str_id in str_ids.split(','):
            opt.gpu_ids.append(int(str_id))
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        args = vars(opt)

        print('-------- [INFO] Options --------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))

        expr_dir = os.path.join(opt.ckpt_dir, opt.model)
        utils.mkdir(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(' [INFO] Options\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
        print('------------- END -------------')
        return opt

    def set_arguments(self):
        # training options
        self.parser.add_argument('--dataset', type=str, default='CARS', help='name of dataset. MNIST default')
        self.parser.add_argument('--data_dir', type=str, default='data', help='root data dir')
        self.parser.add_argument('--small_batch_size', type=int, default=60, help='small batch size')
        self.parser.add_argument('--large_batch_size', type=int, default=100, help='large batch size')
        self.parser.add_argument('--num_preprocess_workers', type=int, default=2, help='num preprocess workers')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids ex) 0,1,2')
        self.parser.add_argument('--ckpt_dir', type=str, default='./ckpt/', help='checkpoint dir')
        self.parser.add_argument('--model', type=str, default='inception', help='name of model')
        self.parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
        self.parser.add_argument('--large_batch_epoch', type=int, default=250, help='large batch epoch')
        self.parser.add_argument('--width_size', type=int, default=299, help='image width size')
        self.parser.add_argument('--channels', type=int, default=3, help='channels')
        self.parser.add_argument('--label_size', type=int, default=98, help='label size')
        self.parser.add_argument('--vector_size', type=int, default=196, help='vector size')
        self.parser.add_argument('--fine_tune', type=int, default=1, help='is fine tune')
        self.parser.add_argument('--eps', type=float, default=1e-8, help='eps')
        self.parser.add_argument('--dropout_rate', type=float, default=0.3, help='dropout rate')
        # visualize options
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8096, help='visdom port of the web display')


class TrainingOptionParser(OptionParser):
    def set_arguments(self):
        super(TrainingOptionParser, self).set_arguments()
        self.parser.add_argument('--is_train', type=int, default=1, help='is training')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                                 help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--print_small_batch', type=int, default=0, help='print small batch for debugging')
        self.parser.add_argument('--plot_freq', type=int, default=15000, help='iteration count per a single plot')


class TestingOptionParser(OptionParser):
    def set_arguments(self):
        super(TestingOptionParser, self).set_arguments()
        self.parser.add_argument('--is_train', type=int, default=0, help='is training')
        self.parser.add_argument('--save_as_numpy', type=int, default=1, help='save as numpy format')
        self.parser.add_argument('--k', type=int, default=10, help='k nearest neighbors.')
        self.parser.add_argument('--test_dir', type=str, default='./test/', help='test dir')

    def parse_args(self):
        opt = self.parser.parse_args()
        str_ids = str(opt.gpu_ids)
        opt.gpu_ids = []
        for str_id in str_ids.split(','):
            opt.gpu_ids.append(int(str_id))
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        args = vars(opt)

        print('-------- [INFO] Options --------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))

        expr_dir = os.path.join(opt.ckpt_dir, opt.model)
        utils.mkdir(expr_dir)
        test_dir = os.path.join(opt.test_dir, opt.model)
        utils.mkdir(test_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(' [INFO] Options\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
        print('------------- END -------------')
        return opt
