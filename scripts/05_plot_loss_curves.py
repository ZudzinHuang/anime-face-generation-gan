import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def save_dcgan_plot(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(df['epoch'], df['g_loss'], marker='o', linewidth=1.8, label='Generator loss')
    ax.plot(df['epoch'], df['d_loss'], marker='o', linewidth=1.8, label='Discriminator loss')
    ax.set_title('DCGAN Training Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'dcgan_loss_curves.png', dpi=220)
    plt.close(fig)

def save_wgan_plot(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(df['epoch'], df['g_loss'], marker='o', linewidth=1.8, label='Generator loss')
    axes[0].plot(df['epoch'], df['critic_loss'], marker='o', linewidth=1.8, label='Critic loss')
    axes[0].set_title('WGAN-GP Generator/Critic Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(df['epoch'], df['gradient_penalty'], marker='o', color='tab:green', linewidth=1.8)
    axes[1].set_title('WGAN-GP Gradient Penalty')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Penalty')
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'wgan_gp_loss_curves.png', dpi=220)
    plt.close(fig)

def save_summary_table(dcgan, wgan, out_dir):
    rows = []
    if dcgan is not None and len(dcgan):
        last = dcgan.iloc[-1]
        rows.append(['DCGAN', int(last['epoch']), f"{last['g_loss']:.3f}", f"{last['d_loss']:.3f}", 'n/a'])
    if wgan is not None and len(wgan):
        last = wgan.iloc[-1]
        rows.append(['WGAN-GP', int(last['epoch']), f"{last['g_loss']:.3f}", f"{last['critic_loss']:.3f}", f"{last['gradient_penalty']:.3f}"])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 1.6 + 0.35 * len(rows)))
    ax.axis('off')
    table = ax.table(
        cellText=rows,
        colLabels=['Model', 'Epochs', 'Generator loss', 'Discriminator/Critic loss', 'Gradient penalty'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    ax.set_title('Final Training Metrics Summary', pad=10)
    fig.tight_layout()
    fig.savefig(out_dir / 'training_metrics_summary_table.png', dpi=220)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Plot loss curves for DCGAN and WGAN-GP runs.')
    parser.add_argument('--runs-dir', default='runs')
    parser.add_argument('--out-dir', default='report_figures')
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dcgan_path = runs_dir / 'dcgan' / 'metrics.csv'
    wgan_path = runs_dir / 'wgan_gp' / 'metrics.csv'
    dcgan = pd.read_csv(dcgan_path) if dcgan_path.exists() else None
    wgan = pd.read_csv(wgan_path) if wgan_path.exists() else None

    if dcgan is not None:
        save_dcgan_plot(dcgan, out_dir)
        print(f'Saved {out_dir / "dcgan_loss_curves.png"}')
    else:
        print(f'Skipping DCGAN plot: missing {dcgan_path}')

    if wgan is not None:
        save_wgan_plot(wgan, out_dir)
        print(f'Saved {out_dir / "wgan_gp_loss_curves.png"}')
    else:
        print(f'Skipping WGAN-GP plot: missing {wgan_path}')

    save_summary_table(dcgan, wgan, out_dir)
    if (out_dir / 'training_metrics_summary_table.png').exists():
        print(f'Saved {out_dir / "training_metrics_summary_table.png"}')

    print('Note: DCGAN and WGAN-GP losses use different objectives, so compare each curve within its own model rather than treating the raw loss values as directly comparable.')

if __name__ == '__main__':
    main()
