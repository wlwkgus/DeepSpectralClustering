import numpy as np
import os
import time

"""
This file is originated from junyanz's repository.
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
"""


class Visualizer(object):
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.win_size = opt.display_winsize
        self.name = opt.model
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)

        self.log_name = os.path.join(opt.ckpt_dir, opt.model, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # errors: dictionary of error labels and values
    def plot_current_error(self, epoch, counter_ratio, error):
        assert len(list(error.keys())) == 1
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(error.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([error[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.array(self.plot_data['X']),
            Y=np.array(self.plot_data['Y']).reshape(-1),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'step',
                'ylabel': 'loss'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, iter_count, errors, t):
        message = '[ epoch : %d, iter count: %d, time: %.3f / batch ] ' % (epoch, iter_count, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
