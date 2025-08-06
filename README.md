# Reconstructing He I 10830 Å from Hα Using pix2pixHD

[![DOI](https://img.shields.io/badge/doi-10.3847%2F1538--4357%2Fadc7fc-blue)](https://doi.org/10.3847/1538-4357/adc7fc)

**Marena, M., Li, Q., Wang, H., & Shen, B.**  
*Astrophysical Journal*, **984**(2):99, May 9 2025.  
DOI: [10.3847/1538-4357/adc7fc](https://doi.org/10.3847/1538-4357/adc7fc)  [oai_citation:0‡researchwith.njit.edu](https://researchwith.njit.edu/en/publications/reconstructing-he-i-10830-%C3%A5-images-using-h%CE%B1-images-through-deep-l?utm_source=chatgpt.com) [oai_citation:1‡mgetit.lib.umich.edu](https://mgetit.lib.umich.edu/resolve?ctx_enc=info%3Aofi%2Fenc%3AUTF-8&ctx_id=10_1&ctx_tim=2025-05-18+08%3A22%3A56&ctx_ver=Z39.88-2004&rfr_id=info%3Asid%2Fprimo.exlibrisgroup.com-iop_doaj_&rft.atitle=Reconstructing+He+i+10830+%C3%85+Images+Using+H%CE%B1+Images+through+Deep+Learning&rft.au=Marena%2C+Marco&rft.date=2025-05-09&rft.eissn=1538-4357&rft.genre=article&rft.issn=0004-637X&rft.issue=2&rft.jtitle=The+Astrophysical+journal&rft.pages=99-&rft.pub=The+American+Astronomical+Society&rft.spage=99&rft.stitle=ApJ&rft.volume=984&rft_dat=%3Ciop_doaj_%3Eapjadc7fc%3C%2Fiop_doaj_%3E&rft_id=info%3Adoi%2F10.3847%2F1538-4357%2Fadc7fc&rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Ajournal&svc_dat=viewit&url_ctx_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Actx&url_ver=Z39.88-2004&utm_source=chatgpt.com)

---

## 📖 Introduction

He I 10830 Å is a key diagnostic of the solar chromosphere and corona, but historical observations are sparse compared to the century-long Hα record. We train a pix2pixHD model to **generate synthetic He I 10830 Å full-disk images from Hα filtergrams**, achieving:

- **Full-disk:** CC = 0.867  
- **Active regions:** CC = 0.903  
- **Nonpolar filaments:** CC = 0.844  
- **Polar-crown filaments:** CC = 0.871  
- **Coronal holes:** CC = 0.536

The model also generalizes across observatories and reconstructs early-2000s X-class flare events in He I 10830 Å  [oai_citation:3‡researchwith.njit.edu](https://researchwith.njit.edu/en/publications/reconstructing-he-i-10830-%C3%A5-images-using-h%CE%B1-images-through-deep-l?utm_source=chatgpt.com).

---

## 📂 Repository Layout

```
.
├── README.md                   # this file
├── LICENSE                     # project license
├── requirements.txt            # Python dependencies
├── checkpoints/                # saved model checkpoints
├── data/                       # raw & processed data (Hα/He I images)
├── datasets/                   # dataset configs & splits
├── models/                     # pretrained models & training logs
├── options/                    # training/inference configuration files
├── results/                    # synthetic outputs & example galleries
├── scripts/                    # helper and processing scripts
├── util/                       # utility modules
├── Boxes_cc_transformed.py     # bounding‐box transformation script
├── Ha2He_accuracy.py           # accuracy evaluation script
├── run_engine.py               # main training/inference orchestrator
├── train.py                    # entry point for model training
└── test.py                     # entry point for model evaluation
```

---

## 🛠 Installation

1. Clone this repo:  
   ```bash
   git clone https://github.com/MarenaMarco/HeI10830-from-Ha.git
   cd HeI10830-from-Ha

2.	Create a conda environment and install dependencies:

conda create -n heli10830 python=3.9
conda activate heli10830
pip install -r requirements.txt


3.	(Optional) Build Docker image:

docker build -t heli10830-demo .



⸻

📊 Data
	•	Sample data is in data/sample/ (already tracked with Git LFS).
	•	Full dataset (~10 GB) of NSO/SOLIS He I 10830 Å and NSO/GONG Hα is hosted on Zenodo or via FTP—see below.
	•	How to add your own:
	1.	Place raw Hα .fits or .png in data/full_dataset/Ha/.
	2.	Place corresponding He I 10830 Å in data/full_dataset/HeI_gt/.
	3.	Run python src/preprocess.py --in_dir data/full_dataset/Ha --out_dir data/processed/Ha.

Tip: keep a small “sample” subset in-repo for quick demos; store the full archive externally (Zenodo, Figshare) and link to it here.

⸻

⚙️ Preprocessing

python src/preprocess.py \
  --input_dir data/full_dataset/Ha \
  --output_dir data/processed/Ha \
  --crop 1024 1024 \
  --normalize

This script will:
	•	Crop/pad images to 1024×1024
	•	Normalize pixel values to [–1, 1]
	•	Split into train/val/test

⸻

🎛 Model Training & Inference

Training

python src/train.py \
  --data_root data/processed \
  --name HeI10830_pix2pixHD \
  --model pix2pixHD \
  --niter 100 \
  --niter_decay 100

Inference

python src/infer.py \
  --checkpoint_dir checkpoints/HeI10830_pix2pixHD \
  --input_dir data/processed/Ha/test \
  --output_dir results/HeI_syn


⸻

🖼️ Results

Hα Input	Ground Truth He I 10830 Å	Synthetic He I 10830 Å (ours)
		

See more examples in results/ or GitHub Pages gallery.

⸻

📑 Citation

If you use this code, please cite:

Marena, M., Li, Q., Wang, H., & Shen, B. (2025). Reconstructing He I 10830 Å Images Using Hα Images through Deep Learning.
Astrophysical Journal, 984(2), Article 99. https://doi.org/10.3847/1538-4357/adc7fc  ￼ ￼

@article{Marena2025HeI10830,
  title = {Reconstructing He i 10830 {{\AA}} Images Using Hα Images through Deep Learning},
  author = {Marena, Marco and Li, Qin and Wang, Haimin and Shen, Bo},
  journal = {Astrophysical Journal},
  volume = {984},
  number = {2},
  pages = {99},
  year = {2025},
  doi = {10.3847/1538-4357/adc7fc}
}


⸻

📝 License

This project is licensed under the MIT License. See LICENSE.

⸻

🤝 Acknowledgements
	•	NSO/SOLIS & NSO/GONG for data provision
	•	NVIDIA’s pix2pixHD framework
	•	Astrophysical Journal

⸻

Next Steps & Tips
	1.	Continuous Integration: Add a GitHub Action to run a quick inference on the sample dataset and verify outputs.
	2.	Docker / Binder: Provide a one-click launch via Binder or Docker Hub for users to try the demo.
	3.	Git LFS: Track only the small sample set; link to the full archive externally.
	4.	Documentation Pages: Use GitHub Pages to host a static gallery of results and usage tutorials.
