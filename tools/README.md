# Tools

## Download Pretrained Weights

It's common to initialize from backbone models pre-trained on ImageNet classification tasks. We use [Swin-Tranformer](https://github.com/microsoft/Swin-Transformer) for our experiments.

- [Official Repo](https://github.com/microsoft/Swin-Transformer)
- `convert-pretrained-model-to-d2.py`: Tool to convert Swin Transformer pre-trained weights for D2.

```bash
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth
python tools/convert-pretrained-model-to-d2.py pretrained/swin_tiny_patch4_window7_224_22k.pth pretrained/swin_tiny_patch4_window7_224_22k.pkl
```



## Analyze Model

- Tool to analyze model parameters, flops and speed.
- Choose one of the following tasks: `flop, speed, parameter or structure`

```bash
python tools/analyze_model.py --num-inputs 100 --tasks [flop speed parameter structure] \
    --config-file configs/muses/swin/dgfusion_swin_tiny_bs8_180k_muses_clre.yaml \
    --use-fixed-input-size MODEL.IS_TRAIN False MODEL.IS_ANALYSIS True \
    MODEL.TEST.DEPTH_ON False MODEL.WEIGHTS <path-to-checkpoint>
```