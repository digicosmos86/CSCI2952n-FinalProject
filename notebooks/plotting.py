import numpy as np

def show_mask(mask, ax, color=[0., 1., 0., 0.2]):
    color=np.array(color)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def add_image_and_mask(img, mask, ax):

    ax.imshow(img)
    show_mask(mask, ax)
    ax.set_axis_off()


def add_image_and_masks(
    img,
    ax,
    mask1,
    mask2,
    colormap,
    label1="ground truth",
    label2="model_output",
    label_intersect="intersection",
):
    ax.imshow(img)
    mask_intersect = np.zeros_like(mask1, dtype=bool)
    mask_intersect[mask1 & mask2] = True
    show_mask(mask_intersect, ax, color=colormap[3])
    
    mask1_only = mask1.copy()
    mask1_only[mask_intersect] = False
    show_mask(mask1_only, ax, color=colormap[2])

    mask2_only = mask2.copy()
    mask2_only[mask_intersect] = False
    show_mask(mask2_only, ax, color=colormap[1])
