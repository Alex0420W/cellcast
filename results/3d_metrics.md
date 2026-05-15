# 3D metrics — CellCast v0 vs StratifiedMeanBaseline (held-out test, 456 conditions)

## Overall

| metric | CellCast | baseline | Δ |
|---|---:|---:|---:|
| pcorr_macro | +0.1270 | +0.1784 | -0.0514 |
| spearcorr_macro | +0.1215 | +0.1754 | -0.0539 |
| top50_dir_acc | +0.7081 | +0.7428 | -0.0347 |
| mse | +0.0070 | +0.0068 | +0.0002 |

## Per cell line

| cell_line | metric | CellCast | baseline | Δ |
|---|---|---:|---:|---:|
| A549 | pcorr_macro | +0.0033 | +0.0488 | -0.0455 |
| A549 | spearcorr_macro | +0.0031 | +0.0318 | -0.0286 |
| A549 | top50_dir_acc | +0.7305 | +0.7521 | -0.0216 |
| A549 | mse | +0.0082 | +0.0080 | +0.0002 |
| K562 | pcorr_macro | +0.0054 | +0.0478 | -0.0424 |
| K562 | spearcorr_macro | +0.0047 | +0.0350 | -0.0303 |
| K562 | top50_dir_acc | +0.7332 | +0.7463 | -0.0132 |
| K562 | mse | +0.0083 | +0.0081 | +0.0001 |
| MCF7 | pcorr_macro | -0.0042 | +0.0883 | -0.0925 |
| MCF7 | spearcorr_macro | -0.0046 | +0.0620 | -0.0666 |
| MCF7 | top50_dir_acc | +0.6605 | +0.7299 | -0.0693 |
| MCF7 | mse | +0.0046 | +0.0043 | +0.0003 |

## Per pathway_level_1 (top-50 DEG direction accuracy only, sorted by Δ)

| pathway | n_conditions | CellCast top50 | baseline top50 | Δ |
|---|---:|---:|---:|---:|
| Metabolic regulation | 12 | +0.7033 | +0.6583 | +0.0450 |
| Focal adhesion signaling | 12 | +0.6850 | +0.6917 | -0.0067 |
| Apoptotic regulation | 12 | +0.7533 | +0.7700 | -0.0167 |
| Epigenetic regulation | 108 | +0.7344 | +0.7617 | -0.0272 |
| Cell cycle regulation | 48 | +0.7967 | +0.8283 | -0.0317 |
| Nuclear receptor signaling | 12 | +0.6483 | +0.6817 | -0.0333 |
| Antioxidant | 12 | +0.6950 | +0.7317 | -0.0367 |
| DNA damage & DNA repair | 48 | +0.6338 | +0.6733 | -0.0396 |
| PKC signaling | 12 | +0.7450 | +0.7850 | -0.0400 |
| Neuronal signaling | 12 | +0.5683 | +0.6100 | -0.0417 |
| Tyrosine kinase signaling | 72 | +0.7050 | +0.7489 | -0.0439 |
| TGF/BMP signaling | 12 | +0.6150 | +0.6600 | -0.0450 |
| HIF signaling | 12 | +0.7317 | +0.7767 | -0.0450 |
| JAK/STAT signaling | 48 | +0.6900 | +0.7371 | -0.0471 |
| Protein folding & Protein degradation | 12 | +0.7233 | +0.7733 | -0.0500 |
| Other | 12 | +0.7167 | +0.7833 | -0.0667 |
