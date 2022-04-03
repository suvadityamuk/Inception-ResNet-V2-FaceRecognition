import os
import pandas

from torchvision.io import read_image   
from torch import optim

from tqdm.notebook import tqdm_notebook
from prefetch_generator import BackgroundGenerator
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn

from PIL import Image
from resnet_model import InceptionResnetV2

PATH = '/Users/suvad/Python Works/images' ## Set this to your own folder which stores the dataset images
print(os.listdir(PATH))

train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(299,299), interpolation=transforms.InterpolationMode.BILINEAR), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.ImageFolder('/Users/suvad/Python Works/images/train', transform=train_transforms)

test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(299,299), interpolation=transforms.InterpolationMode.BICUBIC), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_dataset = datasets.ImageFolder('C:/Users/suvad/Python Works/images/validation', transform=test_transforms)

data_test = DataLoader(test_dataset, shuffle=True, batch_size=1)

data_train = DataLoader(train_dataset, shuffle=True, batch_size=5)

def try_gpu_else_cpu():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
device = try_gpu_else_cpu()

print(torch.cuda.memory_summary(device=device, abbreviated=False))

torch.cuda.empty_cache()

torch.backends.cudnn.benchmark = True

torch.manual_seed(1)
torch.cuda.manual_seed(1)

def load_model_from_checkpoint(path):
    res = torch.load(path)
    model = InceptionResnetV2(feature_list_size=7)
    model.load_state_dict(res['model.state_dict'])
    optimizer = optim.Adam(net.parameters(), weight_decay=0.009, amsgrad=True)
    optimizer.load_state_dict(res['optimizer.state_dict'])
    epoch = res['epoch']
    return model, optimizer, epoch

def train_net(train_loader, epochs=2):
    net = InceptionResnetV2(feature_list_size=7).cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=0.009, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    
    running_loss = 0.00
    count = 0
    
    writer = SummaryWriter()
    
    CURRENT_DIRECTORY = os.getcwd()
    EPOCH_DIRECTORY = os.path.join(CURRENT_DIRECTORY, 'resnet-v2-epochs')
    if not os.path.exists(EPOCH_DIRECTORY):
        os.mkdir(EPOCH_DIRECTORY)
    
    for i in range(epochs):
        
        pbar = tqdm_notebook(enumerate(BackgroundGenerator(train_loader), 0),
                    total=len(train_loader))
        start_time = time.time()
        
        CHECKPOINT_PATH = os.path.join(EPOCH_DIRECTORY, f'model_ckpt_epoch{i+1}.pkl')
        
        for j, data in pbar:
            images, labels = data
            if torch.cuda.is_available():
                inp = torch.autograd.Variable(images).cuda()
                targs = torch.autograd.Variable(labels).cuda()
                
            prepare_time = start_time-time.time()

            optimizer.zero_grad()

            output = net(inp)
            loss = loss_fn(output, targs)
            loss.backward()
            optimizer.step()
            count+=1
            
            process_time = start_time-time.time()-prepare_time
            pbar.set_description(f'Efficiency = {process_time/(process_time+prepare_time)}\nEpochs: {i+1}/{epochs}')
            running_loss += loss.item()
            
            writer.add_scalar('Compute Time efficiency (per epoch)', process_time/(process_time+prepare_time), j)
            writer.add_scalar('Training Loss', loss, j)
            
        scheduler.step(loss)
        torch.save({
            "model.state_dict" : net.state_dict(),
            "optimizer.state_dict" : optimizer.state_dict(),
            "epoch":i
        }, CHECKPOINT_PATH)
    
    writer.close()
    return net, optimizer

net, opt = train_net(data_train, epochs=2)