"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 3D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which takes the 2D spatial + 1D temporal equation directly as a 3D problem
"""


import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer
from NSdataset.NS3DWavegen import * #suwei add
from Adam import Adam#suwei add
from torch_frft.frft_module import  frft, frft_shifted      #suwei add
torch.manual_seed(0)
np.random.seed(0)

################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, alpha=1.0): #suwei add alpha=1.0
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = 10#suwei change modes3 to 16
        self.alpha = alpha
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)) 
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)) 
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)) 
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)) 

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        # x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])                             #suwei comment
        x_ft = torch.fft.fftn(x, dim=[-3,-2,-1])                                #suwei add


        x_ft = x_ft.to(dtype=torch.complex64)                                   #suwei add
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        # x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))    #suwei comment
        x = torch.fft.ifftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1))) #suwei add
        return x.real

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1, device="cuda")#suwei add  device="cuda"
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1,device="cuda")#suwei add  device="cuda"

    def forward(self, x):
        x = self.mlp1(x.float())
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic

        self.p = nn.Linear(13, self.width, dtype=float)# input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv3d(self.width, self.width, 1, dtype=float)
        self.w1 = nn.Conv3d(self.width, self.width, 1, dtype=float)
        self.w2 = nn.Conv3d(self.width, self.width, 1, dtype=float)
        self.w3 = nn.Conv3d(self.width, self.width, 1, dtype=float)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)           
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)      
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)           
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)            
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        return x


    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


################################################################
# configs
################################################################

TRAIN_PATH = 'data/ns_data_V100_N1000_T50_1.mat'
TEST_PATH = 'data/ns_data_V100_N1000_T50_2.mat'

ntrain = 30  #1000                               #suwei change 1000 to 1
ntest = 100

modes = 64#8                                    #suwei change to 64
width = 20

batch_size = 1#10                               #suwei change to 1
learning_rate = 0.001
epochs = 500                               #suwei change to 1000
iterations = 10#epochs*(ntrain//batch_size)     #suwei change to 10

path = 'ns_fourier_3d_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path

runtime = np.zeros(2, )
t1 = default_timer()

sub = 1
S = 64 // sub
T_in = 10#10                                                         #suwei change to 10
T = 10 #40 # T=40 for V1e-3; T=20 for V1e-4; T=10 for V1e-5;            #suwei change to 10

################################################################
# load data
################################################################

# reader = MatReader(TRAIN_PATH)                                      #suwei comment
# train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]         #suwei comment
# train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]   #suwei comment

# reader = MatReader(TEST_PATH)                                       #suwei comment
# test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in]          #suwei comment
# test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]    #suwei comment

a, u = data_read()#data_gen()
start, end = 2,3
train_a = a[-ntrain:,:,:,:T_in]                                        #suwei add
train_u = a[-ntrain:,:,:,T_in:T_in+T]                                  #suwei add
test_a = a[:ntest,:,:,:T_in]                                         #suwei add
test_u = a[:ntest,:,:,T_in:T_in+T]                                   #suwei add
print(train_u.shape)
# print(test_u.shape)                                                 #suwei comment
# assert (S == train_u.shape[-2])                                     #suwei comment
# assert (T == train_u.shape[-1])                                     #suwei comment


a_normalizer = UnitGaussianNormalizer(train_a)                      
train_a = a_normalizer.encode(train_a)                              
test_a = a_normalizer.encode(test_a)                                #suwei comment

y_normalizer = UnitGaussianNormalizer(train_u)                       
train_u = y_normalizer.encode(train_u)                               

train_a = train_a.reshape(ntrain,S,S,1,T_in).repeat([1,1,1,T,1])        
test_a = test_a.reshape(ntest,S,S,1,T_in).repeat([1,1,1,T,1])      #suwei comment

# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)   #suwei comment
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u.reshape(ntrain,S,S,T)), batch_size=batch_size, shuffle=True)     #suwei add
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)     #suwei comment

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

################################################################
# training and evaluation
################################################################
model = FNO3d(modes, modes, modes, width).cuda()
print(count_params(model))
#ptimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)   #suwei comment
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)#suwei add     #suwei add
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
scheduler_step = 20#suwei add
scheduler_gamma = 0.5
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)#suwei add

myloss = LpLoss(size_average=False)
y_normalizer.cuda()                                                      
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x.double()).view(batch_size, S, S, T)

        mse = F.mse_loss(out, y, reduction='mean')
        # mse.backward()

        y = y_normalizer.decode(y)       
        out = y_normalizer.decode(out)   
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()
        
        optimizer.step()
        scheduler.step()#suwei: from original code
        train_mse += mse.item()
        train_l2 += l2.item()
        

    model.eval()
    #suwei comment begin
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x.double()).view(batch_size, S, S, T)
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
    #suwei comment end
    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest#suwei comment

    t2 = default_timer()
    # print(ep, t2-t1, train_mse, train_l2, test_l2)#suwei comment
    print(ep, t2-t1, train_mse, train_l2, test_l2)#suwei add

# torch.save(model, path_model)
#suwei comment begin
pred = torch.zeros(test_u.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.cuda(), y.cuda()

        out = model(x.double()).view(batch_size, S, S, T)
        out = y_normalizer.decode(out)
        pred[index] = out

        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        print(index, test_l2)
        index = index + 1
#suwei comment end
# scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})         #suwei comment