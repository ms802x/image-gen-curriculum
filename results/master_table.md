| Day | Model | Params(M) | Train(s) | VRAM(GB) | Steps | Step(ms) | Recon | FeatDiv | Quality |
|---|---|---|---|---|---|---|---|---|---|
| 1 | baseline VAE          | 2.96          | 26.8      | 0.23     | 1 | 0.45      | 230        | —     | ?/10 |
| 1 | VAE v1 (bigger+anneal)| 11.82     | 25.7   | 0.43  | 1 | 0.44   | 204     | —     | ?/10 |
| 1 | VAE v2 (+perceptual)  | 11.82     | 40.0   | 0.89  | 1 | 0.44   | 75 | —     | ?/10 |
| 1 | baseline GAN (BCE)    | 1.91     | 44.0      | 0.57     | 1 | 0.39      | —     | 0.697 | ?/10 |
| 1 | GAN v1 (stability)    | 1.91 | 69.4  | 0.61 | 1 | 0.39  | —     | 0.657  | ?/10 |
| 1 | GAN v2 (hinge+R1+DiffAug)| 6.58| 86.8   | 0.82  | 1 | 0.40   | —     | 0.674   | ?/10 |
