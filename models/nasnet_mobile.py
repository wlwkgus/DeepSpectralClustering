from collections import OrderedDict

from models.base_model import BaseModel
import torch
from torch import nn
import pretrainedmodels

from utils.utils import tensor2im


class NasnetMobile(BaseModel):
    def __init__(self, opt):
        super(NasnetMobile, self).__init__(opt)
        self.gpu_ids = opt.gpu_ids
        # Model
        self.pre_fc = pretrainedmodels.__dict__['nasnetamobile'](num_classes=1000, pretrained='imagenet')
        if self.gpu_ids:
            self.pre_fc.cuda()
        self.pre_fc.eval()
        self.device = torch.device("cpu")
        self.fc = nn.Linear(
            in_features=51744,
            out_features=opt.vector_size
        )
        self.fc.weight.data.uniform_(-3., 3.)
        if self.gpu_ids:
            self.device = torch.device("cuda")
            self.fc.to(self.device)

        # Tensor
        self.input = self.Tensor(
            opt.small_batch_size,
            opt.channels,
            opt.width_size,
            opt.width_size
        )
        self.label = self.LabelTensor(
            opt.small_batch_size,
            opt.label_size
        )
        self.result = None
        self.c = None
        self.loss_function = nn.L1Loss()
        self.losses = list()
        self.prev_large_batch_loss = 0.

        # Optimizer
        self.large_batch_size = opt.large_batch_size
        self.large_batch_total_epoch = opt.large_batch_epoch
        self.small_batch_count = 0
        self.large_batch_epoch = 0

        if opt.fine_tune:
            params = self.fc.parameters()
        else:
            params = self.pre_fc.parameters() + self.fc.parameters()
        self.optimizer = torch.optim.SGD(
            params=params,
            lr=self.opt.lr
        )

    @property
    def name(self):
        return 'Inception'

    def forward(self, inference=False):
        if self.opt.fine_tune:
            with torch.no_grad():
                pre_fc = self.pre_fc.features(
                    self.input
                )
        else:
            pre_fc = self.pre_fc.features(
                self.input
            )
        pre_fc = pre_fc.view(self.input.size(0), -1)
        if inference:
            with torch.no_grad():
                self.result = self.fc(
                    pre_fc
                )
        else:
            self.result = self.fc(
                pre_fc
            )

    def test(self):
        raise NotImplemented

    def set_input(self, data, is_train=True):
        self.input.resize_(data['img'].size()).copy_(data['img'])
        if is_train:
            self.label.resize_(data['label'].size()).copy_(data['label'])
            self.label = self.label.float()

    def optimize_parameters(self):
        # Look at the paper.
        self.forward()
        # All float tensors
        y_dagger_transpose = self.label / torch.max(
            torch.sum(self.label, dim=0),
            torch.ones(self.label.size(1)).to(self.device)
        )
        self.result = (self.result.t() / self.result.norm(2, 1)).t()
        self.result.contiguous()

        result_dagger = torch.inverse(self.result.t().matmul(self.result)).matmul(self.result.t()).data

        # Objective : This is gradient of result. Make this gradient to zero.
        result_gradient = (self.label - self.result.data.matmul(
            result_dagger.matmul(self.label)
        )).matmul(
            (result_dagger.matmul(y_dagger_transpose)).t()
        )

        loss = torch.trace(self.result.matmul(-result_gradient.t()))
        # print("this is loss : {}".format(loss))
        self.losses.append(loss)
        self.small_batch_count += 1
        if self.small_batch_count >= self.large_batch_size:
            # print('Backwarding... {} / {}'.format(self.large_batch_epoch, self.large_batch_total_epoch))
            loss_sum = None
            for _ in range(self.large_batch_size):
                loss = self.losses.pop(0)
                if loss_sum is None:
                    loss_sum = loss
                else:
                    loss_sum += loss
            self.prev_large_batch_loss = loss_sum.item() / self.large_batch_size
            self.optimizer.zero_grad()
            loss_sum.backward()
            self.optimizer.step()
            self.small_batch_count = 0
            self.large_batch_epoch += 1

    def get_loss(self):
        return OrderedDict([
            ('batch_loss', self.prev_large_batch_loss),
            ('none', 0.)
        ])

    def get_visuals(self, sample_single_image=True):
        image_input = tensor2im(self.input, sample_single_image=sample_single_image)
        return OrderedDict([('input', image_input)])

    def save(self, epoch):
        self.save_network(self.fc, 'fc', epoch, self.gpu_ids)

    def remove(self, epoch):
        raise NotImplemented

    def load(self, epoch):
        self.load_network(self.fc, 'fc', epoch)
