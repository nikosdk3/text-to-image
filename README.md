<img src="./imagen.png" width="450px"></img>

## Text to Image Neural Network (WIP)

Implementation of Google's <a href="https://imagen.research.google/">Imagen</a> in PyTorch

By architecture design, it is much simpler than DALL-E2. It uses a cascading DDPM on text embeddings which are generated from a pretrained T5 LLM. It also improves the clipping for classifier-free guidance, as well as noise level conditioning to make the model aware of the noise added in each step.

This process was improvement from DALL-E2 because instead of using CLIP and a prior network, Imagen uses strong embeddings from a pretrained model like T5 and a diffusion model to generate from these embeddings.

## Citations

```bibtex
@misc{saharia2022photorealistictexttoimagediffusionmodels,
      title={Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding},
      author={Chitwan Saharia and William Chan and Saurabh Saxena and Lala Li and Jay Whang and Emily Denton and Seyed Kamyar Seyed Ghasemipour and Burcu Karagol Ayan and S. Sara Mahdavi and Rapha Gontijo Lopes and Tim Salimans and Jonathan Ho and David J Fleet and Mohammad Norouzi},
      year={2022},
      eprint={2205.11487},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2205.11487},
}
```
