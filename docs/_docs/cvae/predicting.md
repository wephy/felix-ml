---
title: Predicting Diffraction Patterns
category: CVAE
order: 2
---

<head>
  <link rel="stylesheet" href="../../../css/syntax.css" />
</head>

Results below were trained on the dataset intended for CVAE use (see [here](https://wephy.github.io/felix-ml#datasets) for more information on the datasets used). The architecture of the CVAE was extremely straightforward consisting of just a couple of linear layers.

<figure>
    <img src="/felix-ml/assets/images/sample_8.png"
         alt="predictions">
    <figcaption><b>Figure 1</b>&emsp;Predictions for different lattice structures: (top) Varying lattice structures; (middle) Felix generated diffraction pattern; (bottom) CVAE prediction via decoding the lattice.</figcaption>
</figure>

The image was generated with the following code:

```py
# Get a batch of felix_patterns and lattices from the test set
for (felix_patterns, lattices) in test_loader:
    felix_patterns, lattices = felix_patterns.to(device), lattices.to(device)

    # Get 8 (<= batch_size) of each for the image
    felix_patterns, lattices, = felix_patterns[:8], lattices[:8]
    
    # Predict the diffraction pattern based off of the lattice
    prediction_patterns = torch.cat(
        [model.decode(torch.randn(1, latent_size).to(device), l).cpu() for l in lattices])
    
    # Combine all pieces using cat and save
    comparison = torch.cat(
        [lattices.view(-1, 1, 128, 128)[:8],
         felix_patterns.view(-1, 1, 128, 128)[:8],
         prediction_patterns.view(-1, 1, 128, 128)[:8]])
    save_image(comparison.cpu(), 'results/sample_' + str(model.epoch) + '.png')
```
