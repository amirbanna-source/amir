# Archived custom conditional GAN work

These notebooks record the earlier custom implementation, replaced in the active workflow by official StyleGAN2-ADA. They are retained as experimental evidence for the thesis.

| Notebook | Historical purpose |
| --- | --- |
| [09 — Conditional GAN pilot](09_isic2019_binary_cgan_pilot.ipynb) | Executed 5,000-step pilot; generated pool withheld from classifier augmentation after artifact/collapse observations |
| [10 — Generator mode diagnostic](10_isic2019_gan_mode_diagnostic.ipynb) | Saved-generator training/inference-mode comparison; no new fitting |

Code cells and recorded outputs were preserved during the move. Relative documentation links were repaired where needed. Rerunning notebook 09 repeats the old experiment; it does not resume it or prepare the new GAN.

- [Pilot results](../../docs/binary-cgan-pilot-results.md)
- [Mode diagnostic results](../../docs/gan-mode-diagnostic.md)
- [Current notebook index](../README.md)
- [Active StyleGAN2-ADA pilot](13_isic2019_stylegan2ada_pilot.ipynb)

Supporting historical scripts and reports remain in their original repository locations. Training images and image-containing ZIP packages must remain off GitHub.

## Superseded StyleGAN2-ADA pilot

[13 — Old 20-kimg pilot](13_isic2019_stylegan2ada_pilot.ipynb) is archived as of 17 November 2026. The active workflows are [14 — full GAN](../14_isic2019_real_plus_gan_local.ipynb) and [15 — full GAN + conventional](../15_isic2019_real_plus_gan_conventional_local.ipynb). They have no pilot-training stage.
