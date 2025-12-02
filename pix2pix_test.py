import torch
from torchvision import transforms
from dataset import DatasetFromFolder
from model import Generator
import utils
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='facades', help='input dataset')
parser.add_argument('--direction', required=False, default='BtoA', help='input and target image order')
parser.add_argument('--batch_size', type=int, default=1, help='test batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=1024, help='input size')
params = parser.parse_args()
print(params)

# device (CPU or GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directories for loading data and saving results
data_dir = os.path.join('..', 'Data', params.dataset) + os.sep
save_dir = params.dataset + '_test_results' + os.sep
model_dir = params.dataset + '_model' + os.sep

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)

# Data pre-processing
test_transform = transforms.Compose([
    transforms.Resize(params.input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Test data
test_data = DatasetFromFolder(data_dir, subfolder='test', direction=params.direction, transform=test_transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=params.batch_size,
                                               shuffle=False)

# Load model
G = Generator(3, params.ngf, 3).to(device)
# load with map_location so it works on CPU-only installs
state_path = os.path.join(model_dir, 'generator_param.pkl')
if not os.path.exists(state_path):
    raise FileNotFoundError(f"Generator state not found: {state_path}")
G.load_state_dict(torch.load(state_path, map_location=device))
G.eval()

# Test
for i, (input, target) in enumerate(test_data_loader):
    # input & target image data (keep originals on CPU for utils plotting)
    x_ = input.to(device)

    with torch.no_grad():
        gen_image = G(x_)
    gen_image = gen_image.detach().cpu()

    # Show/result for test data
    # utils.plot_test_result expects CPU tensors (batch-first). Pass original input,target (CPU)
    utils.plot_test_result(input, target, gen_image, i, training=False, save=True, save_dir=save_dir)

    print('%d images are generated.' % (i + 1))
