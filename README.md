# 649-project
# Music Genre Classification with Classical Pattern Recognition

This project explores classical pattern recognition techniques (LDA, QDA, SVM) for music genre classification using the GTZAN dataset. It was developed as part of a graduate course in Pattern Recognition (ECEN 649).


Dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download


# Project Structure
649-project/
├── code/
│   ├── extract_features.py         # audio preprocessing and feature extraction
│   ├── train_models.py             # training, evaluation, PCA, result generation
│
├── data/
│   └── genres/                     # raw GTZAN audio files (.wav), 10 genre folders, 100 .wav files in each folder
│
├── features/
│   └── features.csv                # extracted and cleaned audio features
│
├── reports/
│   ├── 649_checkpoint_report.tex   # NeurIPS-style LaTeX report
│   ├── neurips_2025.sty            # official NeurIPS formatting style file
│   ├── neurips_2025.tex            # NeurIPS LaTeX template (unused)
│   └── pca_plot.png                # PCA visualization figure, added here to be seen in report
│
├── results/
│   ├── classifier_accuracies.csv   # accuracy (more decimals)
│   ├── classifier_summary.csv      # accuracy, std dev, prediction counts
│   ├── fold_scores.csv             # 5-fold accuracy results
│   └── pca_plot.png                # PCA visualization figure
│
├── .gitignore
├── README.md
└── requirements.txt                # python dependencies


# Requirements
- Python 3.10+
- `scikit-learn`, `librosa`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`

Install with:
```bash
conda env create -f environment.yml
conda activate 649proj