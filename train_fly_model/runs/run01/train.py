#%%
import yaml
import torch
import numpy as np
import logging
from fly_organelles.run import run
from fly_organelles.model import StandardUnet

from fly_organelles.run import set_weights

logger = logging.getLogger(__name__)

#Q1
log_dir = "/nrs/cellmap/aurrecoecheas/tensorboard/panc_organelle_run01"

CHECKPOINT_PATH = "/nrs/saalfeld/heinrichl/fly_organelles/run08/model_checkpoint_438000"
OLD_CHECKPOINT_CHANNELS = ["all_mem", "organelle", "mito", "er", "nuc", "pm", "vs", "ld"]
labels = ['all_mem', 'mito', 'ld', 'lyso', 'perox', 'isg', 'er','nuc']
else_map = {"perox":"organelle","isg":"organelle","lyso":"organelle"}

yaml_file = "/groups/espinosamedina/home/aurrecoecheas/jrc-muss-pancreas-organelle/train_fly_model/yamls/dataset_generated_newcrops.yaml"
# Q2: "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train/fly_run/datasets_generated_all.yaml"
iterations = 1000000

label_weights = [1,1 ,1,1,1,1]
voxel_size = (8, 8, 8)
l_rate = 0.5e-5
batch_size = 14


if not label_weights:
    label_weights = [
        1.0 / len(labels),
    ] * len(labels)
else:
    if len(label_weights) != len(labels):
        msg = (
            f"If label weights are specified ({type(label_weights)}),"
            f"they need to be of the same length as the list of labels ({len(labels)})"
        )
        raise ValueError(msg)
    normalizer = np.sum(label_weights)
    label_weights = [lw / normalizer for lw in label_weights]
logger.info(
    f"Running training for the following labels:"
    f"{', '.join([f'{lbl} ({lblw:.4f})' for lbl,lblw in zip(labels,label_weights)])}"
)
with open(yaml_file, "r") as data_yaml:
    datasets = yaml.safe_load(data_yaml)
# label_stores, raw_stores, crop_copies = read_data_yaml(data_yaml)

model = StandardUnet(len(labels))
checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True, map_location=torch.device('cpu'))
updated_checkpoint = set_weights(model, checkpoint["model_state_dict"], OLD_CHECKPOINT_CHANNELS, labels, else_map)
model.load_state_dict(updated_checkpoint, strict=True)
model = model.cuda()
run(model,
iterations, 
labels, 
label_weights, 
datasets,
voxel_size = voxel_size,
batch_size = batch_size, 
l_rate=l_rate,
log_dir=log_dir)

