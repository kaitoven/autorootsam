# SAM2 config files

This repo expects a SAM2.1 model config YAML when using `--backbone sam2`.

If you installed the official repository:

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

You can copy the config from that repo, for example:

- `sam2/configs/sam2.1/sam2.1_hiera_l.yaml`

and place it here as:

- `configs/sam2.1/sam2.1_hiera_l.yaml`

Then run:

```bash
python train_timelapse.py --backbone sam2 --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml --sam2_ckpt checkpoints/sam2.1_hiera_large.pt --tile 1024
```
