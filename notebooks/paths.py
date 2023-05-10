from pathlib import Path

home = Path("..")
data_dir = home / "data"
raw_img_dir = data_dir / "images"
truecolor_dir = raw_img_dir / "truecolor"
reflectance_dir = raw_img_dir / "reflectance"

output_train_dir = data_dir / "train"
output_ann_dir = data_dir / "ann"
output_labeled_dir = data_dir / "labeled"

test_dir = data_dir / "test"
gt_dir = data_dir / "ground_truths"
ift_dir = data_dir / "ift_old"