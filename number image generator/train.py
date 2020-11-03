import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator,initialize_weight
import time
#Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
#learning rate
lr = 2e-4
#batch size
bs = 128
image_size = 64
channels_img = 1
z_dim = 100
num_epoch = 5
features_disc = 64
features_gen = 64

# setting transform for dataset
transform = transforms.Compose(
    [ transforms.Resize(image_size),
     transforms.ToTensor(),
     transforms.Normalize(
         [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
     )]
)
#loading dataset
dataset = datasets.MNIST(root='dataset',train = True,transform = transform,download =True)
loader = DataLoader(dataset,batch_size = bs,shuffle = True)

#initialize models
gen = Generator(z_dim,channels_img,features_gen).to(device)
disc = Discriminator(channels_img,features_disc).to(device)
#initialize weights of model
initialize_weight(gen)
initialize_weight(disc)

#initialize optimizer
opt_gen = optim.Adam(gen.parameters(),lr=lr,betas=(0.5,0.999))
opt_disc = optim.Adam(disc.parameters(),lr=lr,betas=(0.5,0.999))
#initialize loss fuction
criterion = nn.BCELoss()

#generatiing fixed noise for better visualzization
fixed_noise = torch.randn(32,z_dim,1,1).to(device)

#writer for tensorboard
writer_real = SummaryWriter(f'logs/real')
writer_fakes = SummaryWriter(f'logs/fake')
step = 0

gen.train()
disc.train()

#training DCGAN
for epoch in range(num_epoch):
    for batch_idx,(real,_) in enumerate(loader):
        start = time.time()
        real = real.to(device)
        noise = torch.randn((bs,z_dim,1,1)).to(device)
        fake = gen(noise)
        ##train discriminator
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real,torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake,torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_fake + loss_disc_real) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph = True)
        opt_disc.step()
        
        ## Train generator 
        
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output,torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch}/{num_epoch} Batch: {batch_idx} Device: {device} Time: {(time.time() - start)}")
            with torch.no_grad():
                fake = gen(fixed_noise)
                
                img_grid_real = torchvision.utils.make_grid(real[:32],normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32],normalize=True)
                
                writer_real.add_image("Real",img_grid_real,global_step = step)
                writer_fakes.add_image("Fake",img_grid_fake,global_step = step)
                
            step+=1
        
        