# mmlatch
Code for MMLatch: Bottom-up Top-down Fusion for Multimodal Sentiment Analysis

[ICASSP 2022 paper](https://ieeexplore.ieee.org/document/9746418)

[preprint](https://arxiv.org/abs/2201.09828)

# Description

Current deep learning approaches for multimodal fusion rely on bottom-up fusion of high and mid-level latent modality representations (late/mid fusion) or low level sensory inputs (early fusion). Models of human perception highlight the importance of top-down fusion, where high-level representations affect the way sensory inputs are perceived, i.e. cognition affects perception. These top-down interactions are not captured in current deep learning models. In this work we propose a neural architecture that captures top-down cross-modal interactions, using a feedback mechanism in the forward pass during network training. The proposed mechanism extracts high-level representations for each modality and uses these representations to mask the sensory inputs, allowing the model to perform top-down feature masking. We apply the proposed model for multimodal sentiment recognition on CMU-MOSEI. Our method shows consistent improvements over the well established MulT and over our strong late fusion baseline, achieving state-of-the-art results.


# How to run

Set PYTHONPATH

```
export PYTHONPATH=$(pwd)/cmusdk:$(pwd)
```

Run your config

```
python run.py --config config.yaml
```

# Citation

If you use this code please cite

```
@INPROCEEDINGS{9746418,
  author={Paraskevopoulos, Georgios and Georgiou, Efthymios and Potamianos, Alexandras},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Mmlatch: Bottom-Up Top-Down Fusion For Multimodal Sentiment Analysis}, 
  year={2022},
  volume={},
  number={},
  pages={4573-4577},
  doi={10.1109/ICASSP43922.2022.9746418}
}
```
