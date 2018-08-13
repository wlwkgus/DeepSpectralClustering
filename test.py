import time
from data.data_loader import get_data_loader
from models.models import create_model
from option_parser import TestingOptionParser
from utils.visualizer import Visualizer
import numpy as np
import torch
from tqdm import tqdm
import os

parser = TestingOptionParser()
opt = parser.parse_args()

data_loader = get_data_loader(opt)

model = create_model(opt)
visualizer = Visualizer(opt)
max_int = 999999999
large_batch_clock = time.time()
model.load(opt.large_batch_epoch)

fs = None
labels = None

# Inference Model
for i, data in enumerate(data_loader):
    # data : dict
    model.set_input(data, is_train=False)
    model.forward(inference=True)
    if fs is None:
        fs = model.result.to(torch.device("cpu")).data.numpy()
        labels = data['label'].numpy()
    else:
        fs = np.concatenate([fs, model.result.to(torch.device("cpu")).data.numpy()])
        labels = np.concatenate([labels, data['label'].numpy()])

# Save as numpy format.
test_dir = os.path.join(opt.test_dir, opt.model)
if opt.save_as_numpy:
    np.save(os.path.join(test_dir, 'labels.npy'), labels)
    np.save(os.path.join(test_dir, 'vectors.npy'), fs)

embeddings = torch.from_numpy(fs).cuda()
# normalize vectors
embeddings = embeddings / embeddings.norm(dim=1).view(-1, 1)
final_index_table = None

for i in tqdm(range(embeddings.size(0))):
    indexes = torch.topk(-torch.sum((embeddings - embeddings[i]) ** 2, 1), k=opt.k+1)[1]

    if final_index_table is None:
        final_index_table = indexes.to(torch.device("cpu")).numpy().reshape([1, -1])
    else:
        final_index_table = np.concatenate([final_index_table, indexes.data.to(torch.device("cpu")).numpy().reshape([1, -1])])

if opt.save_as_numpy:
    np.save(os.path.join(test_dir, 'final_index_table.npy'), final_index_table)

correct = 0
for index_list in final_index_table:
    label_mappings = [labels[index] for index in index_list]
    for mapping in label_mappings[1:]:
        if label_mappings[0] == mapping:
            correct += 1
            break

print("Top k recall @{}".format(opt.k))
print(">>> {} / {} : {}%".format(correct, len(final_index_table), correct / len(final_index_table) * 100))
