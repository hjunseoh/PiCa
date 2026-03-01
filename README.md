# PiCa: Parameter-Efficient Fine-Tuning with Column Space Projection

**[ICLR 2026 Accepted]** Official Implementation of PiCa.

[Optimized version coming soon]

This is the repository for **PiCa (Parameter-efficient Fine-tuning with Column Space Projection)**. For more details, please refer to our paper: [https://arxiv.org/abs/2505.20211](https://arxiv.org/abs/2505.20211).

### Installing Required Packages

```bash
pip install -r requirements.txt
```

### Setting up Commonsense Reasoning
The evaluation datasets (e.g., the `dataset` folder from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters)) are already pre-downloaded into the `LLM-Adapters/ft-training_set` directory.

To train with PiCa:
```bash
cd LLM-Adapters
bash pica.sh
```

To run evaluation:
```bash
bash eval.sh
```

### Setting up Mathematical Reasoning
The MetaMathQA datasets are already pre-downloaded into the `MetaMath/data/train` directory.

To train with PiCa:
```bash
cd MetaMath
bash pica.sh
```

To run evaluation on GSM-8K and MATH:
```bash
bash eval.sh
```

## Acknowledgement
This repository is based on the [SVFT GitHub repository](https://github.com/VijayLingam95/SVFT).

## Citation
```
@article{pica2025,
  title={PiCa: Parameter-Efficient Fine-Tuning with Column Space Projection},
  author={Junseo Hwang and Wonguk Cho and Taesup Kim},
  journal={arXiv preprint arXiv:2505.20211},
  year={2025}
}
```

