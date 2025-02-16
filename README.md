# **Age Transformation with StyleGAN** 👶🏻👴🏻
A deep learning model for realistic age progression using StyleGAN-based transformations.

<img src="outputs/gifs/aging_transformation.gif" width="250" alt="Description of GIF"> <img src="outputs/gifs/aging_transformation2.gif" width="250" alt="Description of GIF"> <img src="outputs/gifs/old_to_young.gif" width="250" alt="Description of GIF">

This project enables **age progression of human faces** using **StyleGAN** and an encoder trained to map real images to the StyleGAN latent space. It provides:  
✅ **Latent Optimization** – Find the best latent representation of a given face.  
✅ **Age Manipulation** – Modify the age of a face using an age boundary.  
✅ **GIF Generation** – Visualize smooth age transitions.  
✅ **Dataset Generation** – Generate synthetic face-latent pairs for training.  

---
## 📂 Project Structure
    .📂 age_transformation/
    │
    ├── 📂 Documentation/              
    │   │── Documentation_German.pdf              
    │   │── Documentation_English.pdf 
    │
    ├── 📂 src/                     # Source code
    │   │── dataset.py              # Dataset handling & downloading
    │   │── latent_encoder.py       # Encoder model & training logic
    │   │── latent_optimization.py  # Optimization of latent vectors
    │   │── face_aging.py           # Age progression logic
    │   └── stylegan_implementation.py # StyleGAN model
    │   
    │── 📂 models/                  # Pretrained models
    │   └── trained_encoder.pt      # Trained encoder weights
    │   
    │── 📂 dataset/                 # Dataset storage
    │   │── images/                 # Generated face images
    │   └── latents/                # Corresponding latent vectors
    │   
    │── 📂 outputs/                 # Generated results
    │   │── gifs/                   # GIFs showing aging progression
    │   └── optimized_faces/        # Images from latent optimization
    │   
    ├── main.py                     # Entry point (CLI interface)  
    ├── LICENSE
    │── .gitignore                  # Ignore unnecessary files
    └── README.md

---
## 🚀 Installation
1️⃣ **Clone this Repository**
```
git clone https://github.com/nickno7/age_transformation.git
cd age_progression
```
2️⃣ **Install Dependencies**
```
pip install -r requirements.txt
```

---
## 🛠️ Usage
The main script main.py provides different functionalities:

1️⃣ **Age a Randomly Generated Face**
```
python main.py --random --alpha 4 --display_gif True
```
✅ Generates a random StyleGAN face, ages it with an intensity of 3 (alpha), and displays the transition.

❗️ **The aging intensity is controlled by the --alpha parameter. A positive value leads to increasing, a negative value to decreasing age.**

2️⃣ **Age Progression of an Input Image**
```
python main.py --image "path/to/image.png" --alpha -5 --display_gif True
```
✅ Ages image.png with an intensity of 5 (alpha) and displays the GIF.

3️⃣ **Generate a New Dataset**
```
python main.py --generate --num_samples=50000
```
✅ Creates a dataset of 50,000 synthetic faces with corresponding latents.

---
## 🧪 How It Works
1️⃣ A face is encoded into the latent space using a ResNet-based encoder.

2️⃣ The latent representation is optimized to match the input image.

3️⃣ The age boundary is applied to manipulate the latent vector.

4️⃣ The modified latent is passed to StyleGAN, generating the aged face.

5️⃣ A GIF is created, showing the transformation over multiple steps.




    



