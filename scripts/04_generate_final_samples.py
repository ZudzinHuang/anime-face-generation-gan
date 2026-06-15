import argparse
from pathlib import Path
import shutil
import torch
import torch.nn as nn
from torchvision import utils

class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=64, nc=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)

def load_generator(checkpoint_path, z_dim, device):
    # weights_only=False is explicit because this checkpoint stores a small dict
    # with model weights plus metadata like epoch number. Only load checkpoints
    # produced by this project.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'G' not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain key 'G'.")
    model = Generator(z_dim=z_dim).to(device)
    model.load_state_dict(checkpoint['G'])
    model.eval()
    return model, checkpoint.get('epoch', 'unknown')

def generate_grid(model, z_dim, n_samples, seed, device, out_path, nrow=8):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    z = torch.randn(n_samples, z_dim, 1, 1, device=device, generator=generator)
    with torch.no_grad():
        images = model(z).detach().cpu()
    utils.save_image(images, out_path, normalize=True, nrow=nrow)

def maybe_copy(src, dst):
    if src.exists():
        shutil.copy2(src, dst)

def main():
    parser = argparse.ArgumentParser(description='Generate clean final sample grids from saved GAN checkpoints.')
    parser.add_argument('--runs-dir', default='runs')
    parser.add_argument('--out-dir', default='report_figures')
    parser.add_argument('--z-dim', type=int, default=100)
    parser.add_argument('--n-samples', type=int, default=64)
    parser.add_argument('--seed', type=int, default=360)
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {
        'dcgan': runs_dir / 'dcgan' / 'latest.pt',
        'wgan_gp': runs_dir / 'wgan_gp' / 'latest.pt',
    }

    for name, ckpt in models.items():
        if not ckpt.exists():
            print(f'Skipping {name}: missing checkpoint {ckpt}')
            continue
        model, epoch = load_generator(ckpt, args.z_dim, device)
        out_path = out_dir / f'{name}_final_generated_samples_seed{args.seed}.png'
        generate_grid(model, args.z_dim, args.n_samples, args.seed, device, out_path)
        print(f'Saved {name} final generated sample grid from epoch {epoch}: {out_path}')

        # Also copy the training snapshot from the final epoch for easy side-by-side report use.
        if isinstance(epoch, int):
            maybe_copy(runs_dir / name / f'samples_epoch_{epoch:03d}.png', out_dir / f'{name}_training_snapshot_epoch_{epoch:03d}.png')

    print('Done. Use the PNGs in report_figures/ for qualitative report figures.')

if __name__ == '__main__':
    main()
