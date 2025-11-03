# Music Genre Classification with CNN

โปรเจคนี้ใช้ Convolutional Neural Network (CNN) ในการจำแนกประเภทเพลงออกเป็น 10 แนวดนตรี ใช้ mel-spectrogram เป็น input feature และสามารถรันบน Google Colab ได้

## Features

- **3 Model Architectures**: MusicCNN, ImprovedMusicCNN, ResNetMusic
- **10 Music Genres**: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- **High Accuracy**: คาดว่าจะได้ 85-95% accuracy
- **Fast Training**: ประมาณ 10-30 นาทีบน GPU
- **Complete Pipeline**: Training, Evaluation, และ Prediction
- **Google Colab Ready**: มี Jupyter Notebook ที่พร้อมใช้งานบน Colab

## Project Structure

```
music-genre-classifier/
├── models/
│   ├── __init__.py
│   └── cnn_model.py          # CNN model architectures
├── utils/
│   ├── __init__.py
│   ├── audio_processor.py    # Audio preprocessing
│   ├── dataset.py            # Dataset และ data loading
│   └── metrics.py            # Evaluation metrics
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── predict.py                # Prediction script
├── Music_Genre_Classification_Complete.ipynb  # Complete Colab notebook
├── requirements.txt
└── README.md
```

## Requirements

```
torch>=2.0.0
torchaudio>=2.0.0
librosa>=0.10.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

## Installation

### Local Installation

```bash
# Clone หรือ download โปรเจค
cd music-genre-classifier

# Install dependencies
pip install -r requirements.txt
```

### Google Colab

วิธีที่ง่ายที่สุดคือใช้ Notebook ที่เตรียมไว้แล้ว:

1. อัปโหลด `Music_Genre_Classification_Complete.ipynb` ไปที่ Google Colab
2. อัปโหลด folder `Data/` ที่มีเพลง 10 แนวไปที่ Colab
3. รันทุก cell ตามลำดับ

**หมายเหตุ**: Notebook นี้มี code ทุกอย่างอยู่ในไฟล์เดียว ไม่ต้องอัปโหลดไฟล์อื่นเพิ่ม!

## Data Format

โครงสร้าง Data folder ควรเป็นแบบนี้:

```
Data/
├── blues/
│   ├── blues.00000.wav
│   ├── blues.00001.wav
│   └── ...
├── classical/
│   ├── classical.00000.wav
│   └── ...
├── country/
├── disco/
├── hiphop/
├── jazz/
├── metal/
├── pop/
├── reggae/
└── rock/
```

แต่ละ folder มีไฟล์เพลง .wav หรือ .mp3 ของแนวนั้นๆ

## Usage

### Training

```bash
python train.py \
  --data_dir /path/to/Data \
  --model MusicCNN \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.001 \
  --save_dir checkpoints
```

**Arguments:**
- `--data_dir`: Path ไปยัง Data folder
- `--model`: เลือก model (`MusicCNN`, `ImprovedMusicCNN`, `ResNetMusic`)
- `--epochs`: จำนวน epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--save_dir`: Directory สำหรับบันทึก checkpoints (default: checkpoints)
- `--early_stopping`: Early stopping patience (default: 10)

**Output:**
- `checkpoints/best_model.pth`: Model ที่ดีที่สุด
- `checkpoints/config.json`: Configuration
- `checkpoints/training_history.png`: กราฟ loss และ accuracy
- `checkpoints/test_results.json`: ผลลัพธ์บน test set

### Evaluation

```bash
python evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --data_dir /path/to/Data \
  --batch_size 32
```

**Output:**
- `evaluation_results/confusion_matrix.png`: Confusion matrix
- `evaluation_results/evaluation_results.json`: Detailed metrics

### Prediction

ทำนายแนวเพลงของไฟล์เดียว:

```bash
python predict.py \
  --checkpoint checkpoints/best_model.pth \
  --audio_path /path/to/song.wav \
  --top_k 3
```

**Example Output:**
```
Predicting genre for: song.wav
================================================================================

Top 3 Predictions:
--------------------------------------------------------------------------------
1. rock           87.34%
2. metal          8.21%
3. pop            2.15%
================================================================================
```

## Model Architectures

### 1. MusicCNN (Recommended)
- Simple และ effective
- 4 conv blocks (32 → 64 → 128 → 256 channels)
- BatchNorm + Dropout สำหรับ regularization
- เหมาะสำหรับ dataset ขนาดกลาง
- Training time: ~10-15 นาที

### 2. ImprovedMusicCNN
- มี residual connections
- Global average pooling
- น้อย parameters กว่า MusicCNN
- Training time: ~15-20 นาที

### 3. ResNetMusic
- ResNet-style architecture
- เหมาะสำหรับ dataset ใหญ่
- มาก parameters มากที่สุด
- Training time: ~20-30 นาที

## Audio Processing

โปรเจคใช้ mel-spectrogram เป็น input feature:

- **Sample Rate**: 22050 Hz
- **Duration**: 3 seconds
- **Mel Bins**: 128
- **FFT Size**: 2048
- **Hop Length**: 512

Audio จะถูก:
1. Resample เป็น 22050 Hz
2. Convert เป็น mono
3. Crop/Pad เป็น 3 วินาที
4. Normalize amplitude
5. แปลงเป็น mel-spectrogram
6. Log scale และ normalize

## Advanced Features (NEW! ⭐)

### 1. Class Weights (แก้ปัญหา Class Imbalance)

ถ้าข้อมูลแต่ละแนวไม่เท่ากัน (เช่น disco 15 เพลง แต่ hip-hop แค่ 2 เพลง):

```bash
python train.py \
  --data_dir /path/to/Data \
  --use_class_weights
```

- Model จะให้ความสำคัญกับ class ที่มีข้อมูลน้อยมากขึ้น
- **แนะนำใช้เสมอ** ถ้าข้อมูลไม่สมดุล

### 2. Data Augmentation (เพิ่มข้อมูลสังเคราะห์)

ถ้าข้อมูลน้อย ใช้ augmentation เพื่อสร้างข้อมูลเพิ่ม:

**Basic Augmentation:**
```bash
python train.py \
  --data_dir /path/to/Data \
  --augmentation basic \
  --augmentation_prob 0.5
```

**SpecAugment (แรงกว่า):**
```bash
python train.py \
  --data_dir /path/to/Data \
  --augmentation specaugment \
  --augmentation_prob 0.5
```

- Time/Frequency masking
- เหมาะสำหรับ dataset เล็ก (< 50 เพลงต่อแนว)

### 3. Focal Loss (โฟกัสที่ hard examples)

ถ้า model เรียนรู้บาง class ยาก:

```bash
python train.py \
  --data_dir /path/to/Data \
  --use_focal_loss \
  --focal_gamma 2.0
```

- Focus มากขึ้นกับ examples ที่ยากจำแนก
- `gamma` สูง = focus มากขึ้น (แนะนำ 1.5-3.0)

### 4. รวมหลายเทคนิค (แนะนำสำหรับ Dataset เล็ก)

```bash
python train.py \
  --data_dir /path/to/Data \
  --model MusicCNN \
  --epochs 100 \
  --use_class_weights \
  --use_focal_loss \
  --augmentation specaugment \
  --augmentation_prob 0.5 \
  --lr 0.0005
```

**คาดว่าจะได้:**
- เพิ่ม accuracy ขึ้น 5-15%
- แก้ปัญหา class ที่เรียนรู้ยาก
- เหมาะสำหรับข้อมูลไม่สมดุลหรือน้อย

## Tips for Better Results

1. **Data Quality**: ใช้ audio ที่มี quality ดี ไม่ corrupted
2. **Class Imbalance**: ใช้ `--use_class_weights` เสมอถ้าข้อมูลไม่เท่ากัน
3. **Small Dataset**: ใช้ `--augmentation specaugment` ถ้ามีข้อมูลน้อย
4. **More Data**: ยิ่งมีข้อมูลเยอะยิ่งดี (แนะนำอย่างน้อย 50-100 เพลงต่อแนว)
5. **Longer Training**: ลอง train นานขึ้นถ้า validation accuracy ยังขึ้นอยู่
6. **Learning Rate**: ลอง adjust learning rate ถ้า loss ไม่ลง (ลอง 0.0005 หรือ 0.0001)
7. **Model Selection**: เริ่มจาก MusicCNN ก่อน ถ้าไม่พอใจค่อยลอง model อื่น

## Troubleshooting

### Out of Memory Error
```bash
# ลด batch size
python train.py --batch_size 16
```

### Audio Loading Error
```bash
# ติดตั้ง ffmpeg
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### Slow Training
- ใช้ Google Colab ที่มี GPU (Runtime → Change runtime type → GPU)
- ลด num_workers ถ้า CPU ไม่แรงพอ
- ใช้ batch_size ที่ใหญ่ขึ้นถ้า memory พอ

## Comparison: Classification vs GAN

| Aspect | GAN (Generation) | CNN (Classification) |
|--------|-----------------|---------------------|
| Difficulty | Very Hard | Easy |
| Training Time | 5-10+ hours | 10-30 minutes |
| Results Quality | Poor (needs 1000+ epochs) | Good (85-95% accuracy) |
| GPU Requirement | Required | Optional but recommended |
| Stability | Unstable | Stable |
| Practical Use | Limited | High |

**คำแนะนำ**: Classification ง่ายและให้ผลดีกว่า GAN มากสำหรับ music tasks!

## Citation

ถ้าใช้ dataset GTZAN:
```
@misc{tzanetakis_essl_cook_2001,
  author = "Tzanetakis, George and Essl, Georg and Cook, Perry",
  title = "Automatic Musical Genre Classification Of Audio Signals",
  year = "2001"
}
```

## License

MIT License - ใช้ได้อย่างอิสระ

## Contact

ถ้ามีปัญหาหรือคำถาม สามารถ:
1. เปิด issue ใน repository
2. ดู documentation เพิ่มเติม
3. ลองดู example ใน Colab notebook

---

**สนุกกับการจำแนกเพลง! 🎵🎸🎹**
