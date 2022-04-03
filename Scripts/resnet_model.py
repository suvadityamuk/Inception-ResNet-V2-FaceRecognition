import torch
from torch import nn
from torch.nn import functional as F

class LambdaScale(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lambda_f = lambda x:x*0.1
    def forward(self, X):
        X = self.lambda_f(X)
        return X

class InceptionResnetv2Stem(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.sub0conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.sub0conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.sub0conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        
        self.sub1p1_mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.sub1p2_conv1 = nn.Conv2d(64, 80, kernel_size=3, stride=2)
        
        self.sub2p1_conv1 = nn.Conv2d(64, 80, kernel_size=1, padding='same')
        self.sub2p1_conv2 = nn.Conv2d(80, 192, kernel_size=3)
        
        self.sub3p2_mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.branch0 = nn.Conv2d(192, 96, kernel_size=1)
        
        self.branch1a = nn.Conv2d(192, 48, kernel_size=1)
        self.branch1b = nn.Conv2d(48, 64, kernel_size=5, padding=2)
        
        self.branch2a = nn.Conv2d(192, 64, kernel_size=1)
        self.branch2b = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.branch2c = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        
        self.branch3a = nn.AvgPool2d(3, padding=1, count_include_pad=False)
        self.branch3b = nn.Conv2d(192, 64, kernel_size=1, stride=1)
        
        self.batchNorm = nn.BatchNorm2d(320)
    
    def forward(self, X):
        
        X = F.relu(self.sub0conv1(X)) 
        X = F.relu(self.sub0conv2(X)) 
        X = F.relu(self.sub0conv3(X)) 
        
        X = self.sub1p1_mpool1(X)
        X = F.relu(self.sub2p1_conv1(X))
        X = F.relu(self.sub2p1_conv2(X))
        
        X = self.sub3p2_mpool1(X)
        
        X0 = self.branch0(X)
        
        X1 = self.branch1a(X)
        X1 = self.branch1b(X1)
        
        X2 = self.branch2a(X)
        X2 = self.branch2b(X2)
        X2 = self.branch2c(X2)
        
        X3 = self.branch3a(X)
        X3 = self.branch3b(X)
        
        X = torch.cat((X0, X1, X2, X3), 1)
        
        X = self.batchNorm(X)
        X = F.relu(X)
        return X

class InceptionResnetv2A(nn.Module):
    def __init__(self, scale=True):
        super().__init__()
        self.scale = scale
        
        self.p1_conv1 = nn.Conv2d(320, 32, kernel_size=1, padding='same')
        
        self.p2_conv1 = nn.Conv2d(320, 32, kernel_size=1, padding='same')
        self.p2_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding='same')
        
        self.p3_conv1 = nn.Conv2d(320, 32, kernel_size=1, padding='same')
        self.p3_conv2 = nn.Conv2d(32, 48, kernel_size=3, padding='same')
        self.p3_conv3 = nn.Conv2d(48, 64, kernel_size=3, padding='same')
        
        self.p_conv1 = nn.Conv2d(128, 320, kernel_size=1, padding='same')
        
        self.batchNorm = nn.BatchNorm2d(320, affine=True)
        
        if self.scale:
            self.scaleLayer = LambdaScale()
        
    def forward(self, X):
        
        # X is relu-activated
        old = X
        
        X1 = F.relu(self.p1_conv1(X))
        
        X2 = F.relu(self.p2_conv1(X))
        X2 = F.relu(self.p2_conv2(X2))
        
        X3 = F.relu(self.p3_conv1(X))
        X3 = F.relu(self.p3_conv2(X3))
        X3 = F.relu(self.p3_conv3(X3))
        
        X = torch.cat((X1, X2, X3), dim=1)
        
        X = self.p_conv1(X)
        if self.scale:
            X = self.scaleLayer(X)
        
        X = self.batchNorm(X)
        X = F.relu(X)
        
        return X

class InceptionResnetv2B(nn.Module):

    def __init__(self, scale=True):
        super().__init__()
        self.scale = scale
        self.p1_conv1 = nn.Conv2d(1088, 192, kernel_size=1, stride=1, padding='same')
        
        self.p2_conv1 = nn.Conv2d(1088, 128, kernel_size=1, padding='same')
        self.p2_conv2 = nn.Conv2d(128, 160, kernel_size=(1,7), padding='same')
        self.p2_conv3 = nn.Conv2d(160, 192, kernel_size=(7,1), padding='same')
        
        self.p3_conv = nn.Conv2d(384, 1088, kernel_size=1, padding='same')
        
        self.batchNorm = nn.BatchNorm2d(1088, affine=True)
        if self.scale:
            self.scaleLayer = LambdaScale()
            
    def forward(self, X):
        old = X
        X1 = F.relu(self.p1_conv1(X))
        
        X2 = F.relu(self.p2_conv1(X))
        X2 = F.relu(self.p2_conv2(X2))
        X2 = F.relu(self.p2_conv3(X2))
        
        X = torch.cat((X1, X2), dim=1)
        
        X = F.relu(self.p3_conv(X))
        if self.scale:
            X = self.scaleLayer(X)
        
        X = self.batchNorm(X)
        X = F.relu(X)
        
        return X

class InceptionResnetv2C(nn.Module):
    def __init__(self, scale=True, noRelu=False):
        super().__init__()
        self.scale = scale
        
        self.noRelu = noRelu
        self.p1_conv1 = nn.Conv2d(2080, 192, kernel_size=1, padding='same')
        
        self.p2_conv1 = nn.Conv2d(2080, 192, kernel_size=1, padding='same')
        self.p2_conv2 = nn.Conv2d(192, 224, kernel_size=(1,3), padding='same')
        self.p2_conv3 = nn.Conv2d(224, 256, kernel_size=(3,1), padding='same')
        
        self.p3_conv = nn.Conv2d(448, 2080, kernel_size=1, padding='same')
        
        self.batchNorm = nn.BatchNorm2d(2080, affine=True)
        if self.scale:
            self.scaleLayer = LambdaScale()
    def forward(self, X):
        old = X
        X1 = F.relu(self.p1_conv1(X))
        
        X2 = F.relu(self.p2_conv1(X))
        X2 = F.relu(self.p2_conv2(X2))
        X2 = F.relu(self.p2_conv3(X2))
        
        X = torch.cat((X1, X2), dim=1)
        
        X = F.relu(self.p3_conv(X))
        if self.scale:
            X = self.scaleLayer(X)
        
        X = self.batchNorm(X)
        if not self.noRelu:
            X = F.relu(X)
        
        return X

class InceptionResnetv2ReductionA(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.p1_mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.p2_conv1 = nn.Conv2d(320, 384, kernel_size=3, stride=2)
        
        self.p3_conv1 = nn.Conv2d(320, 256, kernel_size=1, padding='same')
        self.p3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding='same')
        self.p3_conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=2)
        
        self.batchNorm = nn.BatchNorm2d(1088, affine=True)
        
    def forward(self, X):
        
        X1 = self.p1_mpool1(X)
        
        X2 = F.relu(self.p2_conv1(X))
        
        X3 = F.relu(self.p3_conv1(X))
        X3 = F.relu(self.p3_conv2(X3))
        X3 = F.relu(self.p3_conv3(X3))
        
        X = torch.cat((X1, X2, X3), dim=1)
        
        X = self.batchNorm(X)
        X = F.relu(X)
        
        return X

class InceptionResnetv2ReductionB(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.p1_mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.p2_conv1 = nn.Conv2d(1088, 256, kernel_size=1, padding='same')
        self.p2_conv2 = nn.Conv2d(256, 384, kernel_size=3, stride=2)
        
        self.p3_conv1 = nn.Conv2d(1088, 256, kernel_size=1, padding='same')
        self.p3_conv2 = nn.Conv2d(256, 288, kernel_size=3, stride=2)
        
        self.p4_conv1 = nn.Conv2d(1088, 256, kernel_size=1, padding='same')
        self.p4_conv2 = nn.Conv2d(256, 288, kernel_size=3, padding=1)
        self.p4_conv3 = nn.Conv2d(288, 320, kernel_size=3, stride=2)
        
        self.batchNorm = nn.BatchNorm2d(2080, affine=True)
        
    def forward(self, X):
        
        X1 = self.p1_mpool1(X)
        
        X2 = F.relu(self.p2_conv1(X))
        X2 = F.relu(self.p2_conv2(X2))
        
        X3 = F.relu(self.p3_conv1(X))
        X3 = F.relu(self.p3_conv2(X3))
        
        X4 = F.relu(self.p4_conv1(X))
        X4 = F.relu(self.p4_conv2(X4))
        X4 = F.relu(self.p4_conv3(X4))
        
        X = torch.cat((X1, X2, X3, X4), dim=1)
        
        X = self.batchNorm(X)
        X = F.relu(X)
        
        return X

class InceptionResnetV2(nn.Module):
    def __init__(self, scale=True, feature_list_size=1001):
        super().__init__()
        
        self.scale = scale
        self.stem = InceptionResnetv2Stem()
        self.a = InceptionResnetv2A(scale=True)
        self.b = InceptionResnetv2B(scale=True)
        self.c = InceptionResnetv2C(scale=True)
        self.noreluc = InceptionResnetv2C(scale=True, noRelu=True)
        self.red_a = InceptionResnetv2ReductionA()
        self.red_b = InceptionResnetv2ReductionB()
        
        self.avgpool = nn.AvgPool2d(8)
        
        self.conv2d = nn.Conv2d(2080, 1536, kernel_size=1,)
        
        self.dropout = nn.Dropout(0.8)
        self.flatten = nn.Flatten()
        
        self.linear = nn.Linear(in_features=1536, out_features=feature_list_size)
        
    
    def forward(self, X):
        X = self.stem(X)
        
        for i in range(10):
            X = self.a(X)
        
        X = self.red_a(X)
        
        for i in range(20):
            X = self.b(X)
        
        X = self.red_b(X)
        
        for i in range(9):
            X = self.c(X)
            
        X = self.noreluc(X)
        
        X = self.conv2d(X)
        
        X = self.dropout(X)
        
        X = self.avgpool(X)
        
        X = X.view(X.size(0), -1)
        
        X = self.linear(X)
        
        return X