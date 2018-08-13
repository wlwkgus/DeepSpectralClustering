import time
from data.data_loader import get_data_loader
from models.models import create_model
from option_parser import TrainingOptionParser
from utils.visualizer import Visualizer
import torch

parser = TrainingOptionParser()
opt = parser.parse_args()

data_loader = get_data_loader(opt)

print("[INFO] small batch size : {}".format(opt.small_batch_size))
print("[INFO] large batch size : {}".format(opt.large_batch_size))
print("[INFO] total batch size : {}".format(opt.large_batch_size * opt.small_batch_size))

model = create_model(opt)
visualizer = Visualizer(opt)
max_int = 999999999
large_batch_clock = time.time()
validated_before = list()

for _ in range(max_int):
    for i, data in enumerate(data_loader):
        # data : dict
        small_batch_clock = time.time()
        one_hot_labels = torch.zeros(opt.small_batch_size, opt.label_size, out=torch.LongTensor())
        for j, n in enumerate(data['label']):
            one_hot_labels[j][n.long().data-1] = 1
        data['label'] = one_hot_labels
        model.set_input(data)
        model.optimize_parameters()
        if opt.print_small_batch:
            error = model.get_loss()
            time_delta = time.time() - small_batch_clock
            visualizer.print_current_errors(model.large_batch_epoch, model.small_batch_count, error, time_delta)

        if model.small_batch_count % opt.large_batch_size == 0 and model.large_batch_epoch > 0:
            error = model.get_loss()
            time_delta = time.time() - large_batch_clock
            visualizer.print_current_errors(model.large_batch_epoch, model.small_batch_count, error, time_delta)
            visualizer.plot_current_error(model.large_batch_epoch, 0, error)
            large_batch_clock = time.time()
            if model.lr_scheduler is not None:
                model.lr_scheduler.step()
            model.save(model.large_batch_epoch)

        if model.large_batch_epoch >= opt.large_batch_epoch:
            exit(0)
