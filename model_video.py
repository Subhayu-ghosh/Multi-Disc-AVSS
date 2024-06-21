import torch.nn.functional as F
from torch import nn
from torch.nn import Module
import torch
from audioUtils.hparams import hparams

class MyUpsample(Module):
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(MyUpsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        self.scale_factor = scale_factor if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class VideoGenerator(nn.Module):
    # initializers
    def __init__(self, d=128, dim_neck=32, use_window=True, use_256=False):
        super(VideoGenerator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(256, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, d//2, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d//2)
        if use_256:
            self.deconv6 = nn.ConvTranspose2d(d // 2, d // 4, 4, 2, 1)
            self.deconv6_bn = nn.BatchNorm2d(d // 4)
            self.deconv7 = nn.ConvTranspose2d(d // 4, 3, 4, 2, 1)
        else:
            self.deconv7 = nn.ConvTranspose2d(d // 2, 3, 4, 2, 1)
        if not use_window:
            self.lstm = nn.LSTM(dim_neck*2, 256, 1, batch_first=True)
        else:
            self.window = nn.Conv1d(in_channels=dim_neck*2, out_channels=256, kernel_size=64, stride=4, padding=30)
        self.use_window = use_window
        self.use_256 = use_256

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, return_feature=False):
        # x = F.relu(self.deconv1(input))
        # print(input.shape)
        if self.use_window:
            input = self.window(input.transpose(1,2)).transpose(1,2)
        else:
            input, _ = self.lstm(input)
        # print(input.shape)
        batch_sz, num_frames, feat_dim = input.shape
        input = input.reshape(-1, feat_dim, 1, 1)
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        if self.use_256:
            x = F.relu(self.deconv6_bn(self.deconv6(x)))
        x = torch.tanh(self.deconv7(x))
        x = x.reshape(batch_sz, num_frames, x.shape[1], x.shape[2], x.shape[3])
        if return_feature:
            return x, input
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3d(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        # nn.Upsample(scale_factor=2, mode='nearest'),
        # conv3x3(in_planes, out_planes),
        MyUpsample(scale_factor=(1,2,2), mode='nearest'),
        conv3d(in_planes, out_planes),
        nn.BatchNorm3d(out_planes),
        nn.ReLU(True))
    return block

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class STAGE2_G(nn.Module):
    def __init__(self, residual=False):
        super(STAGE2_G, self).__init__()
        self.STAGE1_G = VideoGenerator()
        # fix parameters of stageI GAN
#         for param in self.STAGE1_G.parameters():
#             param.requires_grad = False
        self.define_module()
        self.residual_video = residual

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(4):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = 32
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        # --> 4ngf x 32 x 32
        self.encoder = nn.Sequential(
            conv3x3(3, ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))
        self.hr_joint = nn.Sequential(
            conv3x3(256 + ngf * 4, ngf * 4),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))
        self.residual = self._make_layer(ResBlock, ngf * 4)
        # --> 2ngf x 64 x 64
        self.upsample1 = upBlock(ngf * 4, ngf * 2)
        # --> ngf x 128 x 128
        self.upsample2 = upBlock(ngf * 2, ngf)
        # --> ngf // 2 x 256 x 256
        self.upsample3 = upBlock(ngf, ngf // 2)
        # --> ngf // 4 x 512 x 512
        self.upsample4 = upBlock(ngf // 2, ngf // 4)
        # --> 3 x 512 x 512
        self.img = nn.Sequential(
            conv3d(ngf // 4, 3),
            nn.Tanh())

    def forward(self, input, train=False):
        stage1_video, audio_embedding = self.STAGE1_G(input, return_feature=True)
        batch_sz, num_frames, _,_,_ = stage1_video.shape
        encoded_frames = self.encoder(stage1_video.reshape(batch_sz*num_frames,3,128,128))

        c_code = audio_embedding.reshape(batch_sz*num_frames,256,1,1)
        c_code = c_code.repeat(1, 1, 32, 32)
        i_c_code = torch.cat([encoded_frames, c_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code) # (bs*num_frame)*4ngf*32*32

        h_code = h_code.reshape(batch_sz, num_frames, -1, 32, 32).transpose(2,1)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)

        stage2_video = self.img(h_code)
        stage2_video = stage2_video.transpose(2,1).reshape(batch_sz, num_frames, 3, 512, 512)

        if self.residual_video:
            stage2_video = MyUpsample(scale_factor=(1,4,4), mode='nearest')(stage1_video) + stage2_video

        if train:
            return stage1_video, stage2_video
        return stage2_video


class UInet(nn.Module):
    def __init__(self, embedding_user, embedding_item, embedding_size=16, out_channels=64, kernel_size=2, stride=1, padding=0, n_class=1):
        super(UInet, self).__init__()
        self.embedding_size = embedding_size
        self.embedding_user, self.embedding_item = embedding_user, embedding_item
        self.cnn = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(int(((self.embedding_size-kernel_size+2*padding)/stride+1)*out_channels), n_class)
 
    def forward(self, x):
        embed_users = self.embedding_user(x[:,0])
        embed_items0 = self.embedding_item(x[:,1])
        embed_items1 = self.embedding_item(x[:,2])
        out = torch.cat([embed_users, embed_items0],1).reshape(-1, 1, 2, self.embedding_size)
        out = self.cnn(out)          
        out = self.relu(out)
        out = torch.flatten(out, 1)
        out = self.linear(out) 
        return out
    
    def predict(self, pairs, batch_size, verbose):
        """Computes predictions for a given set of user-item pairs.
        Args:
          pairs: A pair of lists (users, items) of the same length.
          batch_size: unused.
          verbose: unused.
        Returns:
          predictions: A list of the same length as users and items, such that
          predictions[i] is the models prediction for (users[i], items[i]).
        """
        del batch_size, verbose
        num_examples = len(pairs[0])
        assert num_examples == len(pairs[1])
        predictions = np.empty(num_examples)
        pairs = np.array(pairs, dtype=np.int16)
        for i in range(num_examples):
            x = np.c_[pairs[0][i],pairs[1][i],pairs[1][i]]
            x = torch.from_numpy(x).long()
            out = self.forward(x)
            predictions[i] = out.reshape(-1).data.numpy()
        return predictions

class UIInet(nn.Module):
    def __init__(self, embedding_user, embedding_item, embedding_size=16, out_channels=64, kernel_size=2, stride=1, padding=0, n_class=1):
        super(UIInet, self).__init__()
        self.embedding_size, self.kernel_size = embedding_size, kernel_size
        self.embedding_user, self.embedding_item = embedding_user, embedding_item
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.cnn2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        if self.kernel_size == 2:
            input_size = (self.embedding_size - self.kernel_size + 2 * padding)/stride + 1
            channel_num = out_channels
        else:
            input_size = self.embedding_size
            channel_num = out_channels
        self.linear = nn.Linear(int(((input_size - self.kernel_size + 2 * padding)/stride + 1) * channel_num), n_class)
 
    def forward(self, x):
        embed_users = self.embedding_user(x[:,0])
        embed_items0 = self.embedding_item(x[:,1])
        embed_items1 = self.embedding_item(x[:,2])
        out = torch.cat([embed_items0, embed_users, embed_items1],1).reshape(-1, 1, 3, self.embedding_size)
        out = self.cnn1(out)          
        out = self.relu(out)
        if self.kernel_size == 2: 
            out = self.cnn2(out)          
            out = self.relu(out)
        out = torch.flatten(out, 1)
        out = self.linear(out) 
        return out
    
    def predict(self, pairs, batch_size, verbose):
        """Computes predictions for a given set of user-item pairs.
        Args:
          pairs: A pair of lists (users, items) of the same length.
          batch_size: unused.
          verbose: unused.
        Returns:
          predictions: A list of the same length as users and items, such that
          predictions[i] is the models prediction for (users[i], items[i]).
        """
        del batch_size, verbose
        num_examples = len(pairs[0])
        assert num_examples == len(pairs[1])
        predictions = np.empty(num_examples)
        pairs = np.array(pairs, dtype=np.int16)
        for i in range(num_examples):
            x = np.c_[pairs[0][i],pairs[1][i],pairs[1][i]]
            x = torch.from_numpy(x).long()
            out = self.forward(x)
            predictions[i] = out.reshape(-1).data.numpy()
        return predictions

class Cooccnet(nn.Module):
    def __init__(self, users_num, items_num, embedding_size=16, out_channels=64, kernel_size=2, stride=1, padding=0, n_class=1):
        super(Net, self).__init__()
        self.embedding_size, self.kernel_size, self.items_num, self.users_num = embedding_size, kernel_size, items_num, users_num
        self.embedding_user  = nn.Embedding(self.users_num, self.embedding_size)
        self.embedding_item = nn.Embedding(self.items_num, self.embedding_size)
        #self.embedding_user  = nn.Embedding.from_pretrained(torch.nn.init.normal(tensor=torch.Tensor(self.users_num, self.embedding_size), mean=0, std=0.1))
        #self.embedding_item = nn.Embedding.from_pretrained(torch.nn.init.normal(tensor=torch.Tensor(self.items_num, self.embedding_size), mean=0, std=0.1))

        self.net_ui = UInet(embedding_user=self.embedding_user, 
                            embedding_item=self.embedding_item, 
                            embedding_size=self.embedding_size, 
                            out_channels=out_channels, 
                            kernel_size=2, 
                            stride=stride, 
                            padding=padding, 
                            n_class=n_class)
        self.net_uii = UIInet(embedding_user=self.embedding_user, 
                              embedding_item=self.embedding_item, 
                              embedding_size=self.embedding_size, 
                              out_channels=out_channels, 
                              kernel_size=self.kernel_size, 
                              stride=stride, 
                              padding=padding, 
                              n_class=n_class)
 
    def forward(self, x):
        out1 = self.net_ui(x)          
        out2 = self.net_uii(x) 
        return out1, out2
    
    def predict(self, pairs, batch_size, verbose):
        """Computes predictions for a given set of user-item pairs.
        Args:
          pairs: A pair of lists (users, items) of the same length.
          batch_size: unused.
          verbose: unused.
        Returns:
          predictions: A list of the same length as users and items, such that
          predictions[i] is the models prediction for (users[i], items[i]).
        """
        del batch_size, verbose
        num_examples = len(pairs[0])
        assert num_examples == len(pairs[1])
        predictions = np.empty(num_examples)
        pairs = np.array(pairs, dtype=np.int16)
        for i in range(num_examples):
            x = np.c_[pairs[0][i],pairs[1][i],pairs[1][i]]
            x = torch.from_numpy(x).long()
            out, _ = self.forward(x)
            predictions[i] = out.reshape(-1).data.numpy()
        return predictions
    
    def get_embeddings(self):
        idx = torch.LongTensor([i for i in range(self.items_num)])
        embeddings = self.embedding_item(idx)
        return embeddings
