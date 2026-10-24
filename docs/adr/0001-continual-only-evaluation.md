# Continual protocol is the only evaluation protocol

All reference and deployed runs use the continual protocol (15 corruptions in the standard order, no reset, 3 shuffle seeds). The episodic protocol is dropped. Resetting at corruption boundaries needs oracle knowledge that a deployed edge system never has, and running two protocols doubles the final campaign for no deployment claim.

## Consequences

- Episodic paper results (for example SAR's Table 4) no longer compare.
- The external sanity check is ROID (Marsden et al., WACV 2024, arXiv:2306.00650), Table 1, continual IN-C, ViT-b-16: source 60.2% error, TENT 54.5% error, 5 runs. To make it comparable, the project adopts ROID's checkpoint (torchvision `vit_b_16`, `IMAGENET1K_V1`), its RobustBench 5,000 subset and TENT's default optimizer (SGD, momentum 0.9, Nesterov, lr 2.5e-4 at B = 64). The gap at (B, S) = (64, 1) is reported. ROID's repo is not run.
- TENT at small B is expected to collapse without resets. Such configs are labelled collapsed and reported, never dropped (see GLOSSARY.md).
