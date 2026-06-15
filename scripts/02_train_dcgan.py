import argparse, math, os, csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from tqdm import tqdm

class Generator(nn.Module):
    def __init__(self,z_dim=100,ngf=64,nc=3):
        super().__init__()
        self.net=nn.Sequential(
            nn.ConvTranspose2d(z_dim,ngf*8,4,1,0,bias=False), nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False), nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False), nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf,nc,4,2,1,bias=False), nn.Tanh())
    def forward(self,z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self,ndf=64,nc=3,bn=True):
        super().__init__()
        def block(i,o,norm=True):
            layers=[nn.Conv2d(i,o,4,2,1,bias=False)]
            if norm and bn: layers.append(nn.BatchNorm2d(o))
            layers.append(nn.LeakyReLU(0.2,True))
            return layers
        self.net=nn.Sequential(*block(nc,ndf,False),*block(ndf,ndf*2),*block(ndf*2,ndf*4),*block(ndf*4,ndf*8),nn.Conv2d(ndf*8,1,4,1,0,bias=False))
    def forward(self,x): return self.net(x).view(-1)

def weights_init(m):
    name=m.__class__.__name__
    if 'Conv' in name: nn.init.normal_(m.weight.data,0.0,0.02)
    elif 'BatchNorm' in name:
        nn.init.normal_(m.weight.data,1.0,0.02); nn.init.constant_(m.bias.data,0)

def loader(data,batch_size):
    tfm=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    ds=datasets.ImageFolder(str(Path(data).parent), transform=tfm) if not any(Path(data).glob('*.*')) else datasets.ImageFolder(str(Path(data).parent), transform=tfm)
    # If images are directly in data/, create a temporary class by using parent + folder name with ImageFolder.
    return DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True), len(ds)

def save_grid(G,fixed_noise,path):
    G.eval()
    with torch.no_grad(): fake=G(fixed_noise).detach().cpu()
    utils.save_image(fake,path,normalize=True,nrow=8)
    G.train()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True); ap.add_argument('--epochs',type=int,default=5)
    ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--z-dim',type=int,default=100)
    ap.add_argument('--lr',type=float,default=2e-4)
    args=ap.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    run=Path('runs/dcgan'); run.mkdir(parents=True,exist_ok=True)
    dl,n=loader(args.data,args.batch_size)
    G=Generator(args.z_dim).to(device); D=Discriminator().to(device)
    G.apply(weights_init); D.apply(weights_init)
    optG=optim.Adam(G.parameters(),lr=args.lr,betas=(0.5,0.999)); optD=optim.Adam(D.parameters(),lr=args.lr,betas=(0.5,0.999))
    crit=nn.BCEWithLogitsLoss(); fixed=torch.randn(64,args.z_dim,1,1,device=device)
    rows=[]
    for ep in range(1,args.epochs+1):
        g_sum=d_sum=0; steps=0
        for real,_ in tqdm(dl,desc=f'dcgan epoch {ep}'):
            real=real.to(device); b=real.size(0)
            z=torch.randn(b,args.z_dim,1,1,device=device); fake=G(z)
            optD.zero_grad(); lossD=crit(D(real),torch.ones(b,device=device))+crit(D(fake.detach()),torch.zeros(b,device=device)); lossD.backward(); optD.step()
            optG.zero_grad(); lossG=crit(D(fake),torch.ones(b,device=device)); lossG.backward(); optG.step()
            g_sum+=lossG.item(); d_sum+=lossD.item(); steps+=1
        row={'epoch':ep,'g_loss':g_sum/steps,'d_loss':d_sum/steps,'n_images':n}; rows.append(row); print(row)
        save_grid(G,fixed,run/f'samples_epoch_{ep:03d}.png')
        torch.save({'G':G.state_dict(),'D':D.state_dict(),'epoch':ep},run/'latest.pt')
    with open(run/'metrics.csv','w',newline='') as f: csv.DictWriter(f,fieldnames=rows[0].keys()).writeheader(); csv.DictWriter(f,fieldnames=rows[0].keys()).writerows(rows)
if __name__=='__main__': main()
