# GPU setup note

My original `.venv` was using Python 3.14 with CPU-only PyTorch (`torch 2.12.0+cpu`). Because of that, PyTorch could not use CUDA even though `nvidia-smi` showed that the RTX 3060 Ti was available.

I made a separate GPU environment here:

```text
E:\Documents\Coding\APS360 Project\.venv310
```

This environment uses:

```text
Python 3.10
torch 2.5.1+cu121
torchvision 0.20.1+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 3060 Ti
```

To use it for training, I can run:

```powershell
cd "E:\Documents\Coding\APS360 Project"
.\.venv310\Scripts\Activate.ps1
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Then I can rerun the training scripts with the GPU environment:

```powershell
python scripts\02_train_dcgan.py --data data\processed\anime_faces_64 --epochs 30 --batch-size 128
python scripts\03_train_wgan_gp.py --data data\processed\anime_faces_64 --epochs 30 --batch-size 64
```

For the progress report, the short 5-epoch runs were mainly used to show that the full pipeline works. The longer GPU runs should produce more coherent sample grids.
