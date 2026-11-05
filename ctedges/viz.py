import matplotlib.pyplot as plt
import numpy as np


def show_sequences(original, linked2d, linked3d, names=None, max_cols=8, save=None, show=True):
    """Original slices, slices after 2D linking and after 3D linking, one row each."""
    n = min(len(original), max_cols)
    fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 7), squeeze=False)
    rows = [("original", original, "gray"), ("edges linked in slice", linked2d, "gray"),
            ("linked between slices", linked3d, "gray")]
    for r, (title, seq, cmap) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(seq[c], cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0 and names is not None:
                ax.set_title(names[c], fontsize=8)
        axes[r, 0].set_ylabel(title, fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
    if show:
        plt.show()
    plt.close(fig)


def overlay(base, added, colour=(1.0, 0.75, 0.0)):
    """White = base edges, coloured = pixels added on top of them."""
    rgb = np.zeros(base.shape + (3,))
    rgb[base > 0] = 1.0
    extra = (added > 0) & ~(base > 0)
    rgb[extra] = colour
    return rgb
