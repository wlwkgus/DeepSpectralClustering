from collections import OrderedDict

from models.base_inception import inception_v3_without_last_layer
from models.base_model import BaseModel
import torch
from torch import nn
import torch.nn.functional as F

from utils.utils import tensor2im


class Inception(BaseModel):
    def __init__(self, opt):
        super(Inception, self).__init__(opt)
        self.gpu_ids = opt.gpu_ids
        # Model
        self.pre_fc = inception_v3_without_last_layer(pretrained=True)
        self.device = torch.device("cpu")
        self.pre_fc.eval()
        if self.gpu_ids:
            self.device = torch.device("cuda")
            self.pre_fc.to(self.device)
        self.fc = nn.Linear(
            in_features=2048,
            out_features=opt.vector_size
        )
        self.fc.weight.data.uniform_(-2.0, 2.0)
        self.fc.bias.data.uniform_(-2.0, 2.0)
        if self.gpu_ids:
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
        self.loss_threshold = 5e8

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
                pre_fc = self.pre_fc(
                    self.input
                )
        else:
            pre_fc = self.pre_fc(
                self.input
            )
        if inference:
            with torch.no_grad():
                self.result = self.fc(
                    pre_fc
                )
        else:
            self.result = self.fc(
                pre_fc
            )
        self.result = F.dropout(self.result, p=self.opt.dropout_rate, training=self.opt.is_train)

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
        with torch.no_grad():
            y_dagger_transpose = self.label / torch.max(
                torch.sum(self.label, dim=0),
                torch.ones(self.label.size(1)).to(torch.device("cuda:{}".format(self.opt.gpu_ids[0])))
            )

            result_dagger = torch.inverse(self.result.t().matmul(self.result)).matmul(self.result.t()).data

            # Objective : This is gradient of result. Make this gradient to zero.
            result_gradient = (self.label - self.result.data.matmul(
                result_dagger.matmul(self.label)
            )).matmul(
                (result_dagger.matmul(y_dagger_transpose)).t()
            )

        loss = torch.trace(self.result.matmul(-result_gradient.t()))
        loss /= self.opt.small_batch_size
        self.losses.append(loss)
        self.small_batch_count += 1
        if self.small_batch_count >= self.large_batch_size:
            loss_sum = None
            for _ in range(self.large_batch_size):
                loss = self.losses.pop(0)
                if loss_sum is None:
                    loss_sum = loss
                else:
                    loss_sum += loss
            self.prev_large_batch_loss = loss_sum.item() / self.large_batch_size
            loss_sum /= self.large_batch_size
            if loss_sum <= self.loss_threshold:
                self.optimizer.zero_grad()
                loss_sum.backward()
                self.optimizer.step()
            self.small_batch_count = 0
            self.large_batch_epoch += 1

    def get_loss(self):
        return OrderedDict([
            ('batch_loss', self.prev_large_batch_loss),
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
