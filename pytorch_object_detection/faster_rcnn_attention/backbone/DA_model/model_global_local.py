import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)

class netD_pixel(nn.Module):
    def __init__(self,in_channels=256, context=False):
        """ local alignment """
        super(netD_pixel, self).__init__()
        out_channels=64
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1,
                  padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(1)
        self.context = context
        self._init_weights()

    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    
 
    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # tag: yang adds BN layer
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.context:
          # print('-----------x', x.shape)#  torch.Size([1, 128, 150, 150])
          feat = F.avg_pool2d(x, (x.size(2), x.size(3))) # torch.Size([1, 128, 1, 1])
          # x = self.conv3(x)
          # tag: yang adds BN layer
          x = self.bn3(self.conv3(x))
          # print('-----------feat', feat.shape)
          # print('---------------torch.sigmoid(x)', torch.sigmoid(x).shape) # torch.sigmoid(x) torch.Size([1, 1, 150, 150])
          return torch.sigmoid(x),feat
        else:
          x = self.conv3(x)
          # tag: yang adds BN layer
          # x = self.bn3(self.conv3(x))
          return torch.sigmoid(x)

class netD(nn.Module):
    def __init__(self, in_channels=256, context=False):
        """ global alignment """
        super(netD, self).__init__()
        # self.conv1 = conv3x3(1024, 512, stride=2)
        # # self.bn1 = nn.BatchNorm2d(512)
        # self.conv2 = conv3x3(512, 128, stride=2)
        self.out_chanels = 64
        self.conv2 = conv3x3(in_channels, self.out_chanels, stride=2)
        self.bn2 = nn.BatchNorm2d(self.out_chanels)
        self.conv3 = conv3x3(self.out_chanels, self.out_chanels, stride=2)
        self.bn3 = nn.BatchNorm2d(self.out_chanels)
        self.fc = nn.Linear(self.out_chanels,2)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      # normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        # x = F.dropout(F.relu(self.conv1(x)),training=self.training)
        # x = F.dropout(F.relu(self.conv2(x)),training=self.training)
        # x = F.dropout(F.relu(self.conv3(x)),training=self.training)
        # tag: yang adds BN
        # x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        # x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        # x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.leaky_relu(self.bn2(self.conv2(x)))
        # x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = F.avg_pool2d(x,(x.size(2),x.size(3))) # torch.Size([1, 128, 1, 1])
        # print('-----------x after pooling', x.shape) 
        x = x.view(-1,self.out_chanels) # torch.Size([1, 128])
        # print('-----------x after viewing', x.shape)
        if self.context:
          feat = x
        x = self.fc(x) # torch.Size([1, 2]) tensor([[-0.1457, -0.0657]], device='cuda:0', grad_fn=<ThAddmmBackward>)
        # print('-----------x after fc', x.shape)
        # tag: yang adds
        x = F.softmax(x, dim=1)
        if self.context:
          return x,feat
        else:
          return x


class netD_dc(nn.Module):
    def __init__(self):
        super(netD_dc, self).__init__()
        self.fc1 = nn.Linear(2048,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,2)
        
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))),training=self.training)
        x = self.fc3(x)
        return x