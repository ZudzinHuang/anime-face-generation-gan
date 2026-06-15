import argparse, json, random
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

IMG_EXTS={'.jpg','.jpeg','.png','.webp','.bmp'}

def center_crop_square(img):
    w,h=img.size
    s=min(w,h)
    left=(w-s)//2
    top=(h-s)//2
    return img.crop((left,top,left+s,top+s))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--size',type=int,default=64)
    ap.add_argument('--max-images',type=int,default=0)
    args=ap.parse_args()
    inp=Path(args.input); out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    run=Path('runs/data_prep'); run.mkdir(parents=True,exist_ok=True)
    paths=[p for p in inp.rglob('*') if p.suffix.lower() in IMG_EXTS]
    if args.max_images: paths=paths[:args.max_images]
    kept=0; skipped=0; means=[]; stds=[]
    examples=[]
    for p in tqdm(paths,desc='cleaning'):
        try:
            img=Image.open(p)
            img=ImageOps.exif_transpose(img).convert('RGB')
            img=center_crop_square(img).resize((args.size,args.size),Image.BICUBIC)
            arr=np.asarray(img).astype('float32')/255.0
            means.append(arr.mean(axis=(0,1)).tolist())
            stds.append(arr.std(axis=(0,1)).tolist())
            name=f'{kept:06d}.png'
            img.save(out/name)
            if len(examples)<64: examples.append(arr)
            kept+=1
        except Exception:
            skipped+=1
    stats={
        'raw_files_found':len(paths),'cleaned_images':kept,'skipped_unreadable':skipped,
        'image_size':args.size,
        'mean_rgb':np.array(means).mean(axis=0).round(4).tolist() if means else None,
        'std_rgb':np.array(stds).mean(axis=0).round(4).tolist() if stds else None,
        'processed_dir':str(out)
    }
    (run/'dataset_stats.json').write_text(json.dumps(stats,indent=2))
    if examples:
        n=min(64,len(examples)); cols=8; rows=(n+cols-1)//cols
        fig,axs=plt.subplots(rows,cols,figsize=(cols,rows))
        axs=np.array(axs).reshape(-1)
        for ax in axs: ax.axis('off')
        for ax,arr in zip(axs,examples[:n]): ax.imshow(arr)
        fig.suptitle(f'Cleaned anime face samples, n={kept}')
        fig.tight_layout()
        fig.savefig(run/'cleaned_sample_grid.png',dpi=200)
    print(json.dumps(stats,indent=2))

if __name__=='__main__': main()
