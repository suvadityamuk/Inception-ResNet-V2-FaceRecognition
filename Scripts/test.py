from PIL import Image
from resnet_model import InceptionResnetV2
from torchvision.io import read_image   
from torch import optim
import os
from prefetch_generator import BackgroundGenerator

def try_gpu_else_cpu():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
device = try_gpu_else_cpu()

test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(299,299), interpolation=transforms.InterpolationMode.BICUBIC), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_dataset = datasets.ImageFolder('C:/Users/suvad/Python Works/images/validation', transform=test_transforms)

data_test = DataLoader(test_dataset, shuffle=True, batch_size=1)

def load_model_from_checkpoint(path):
    res = torch.load(path)
    model = InceptionResnetV2(feature_list_size=7)
    model.load_state_dict(res['model.state_dict'])
    optimizer = optim.Adam(net.parameters(), weight_decay=0.009, amsgrad=True)
    optimizer.load_state_dict(res['optimizer.state_dict'])
    epoch = res['epoch']
    return model, optimizer, epoch

def predict_class(img, transform_func):
    classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    var = torch.autograd.Variable(img)
    
    # Use latest model epoch by changing path
    model, opt, ep = load_model_from_checkpoint("C:\\Users\\suvad\\Python Works\\resnet-v2-epochs\\model_ckpt_epoch2.pkl")
    res = model(var)
    res = res.cpu()
    clsf = res.data.numpy().argmax()
    print(clsf)
    pred = classes[clsf]
    return pred

torch.cuda.empty_cache()

for i, data in enumerate(data_test):
    images, labels = data
    predict_class(images, test_transforms)

