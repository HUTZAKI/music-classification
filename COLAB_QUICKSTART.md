# Google Colab Quick Start Guide

คู่มือสำหรับใช้โปรเจคบน Google Colab อย่างง่าย

## วิธีที่ 1: ใช้ Notebook (แนะนำ - ง่ายที่สุด!)

**ไฟล์**: `Music_Genre_Classification_Complete.ipynb`

### ขั้นตอน:

1. **อัปโหลด Notebook**
   - ไปที่ [Google Colab](https://colab.research.google.com)
   - File → Upload notebook
   - เลือก `Music_Genre_Classification_Complete.ipynb`

2. **อัปโหลดข้อมูล**
   - คลิกที่ไอคอน folder ด้านซ้าย
   - อัปโหลด folder `Data/` ที่มีเพลง 10 แนว
   - รอจนอัปโหลดเสร็จ

3. **เปิด GPU**
   - Runtime → Change runtime type
   - Hardware accelerator → GPU
   - Save

4. **รันโค้ด**
   - กด Runtime → Run all
   - หรือกด Shift+Enter ในแต่ละ cell ตามลำดับ

5. **ดูผลลัพธ์**
   - จะได้ model ที่ train เสร็จแล้ว
   - จะได้ confusion matrix
   - จะได้ accuracy และ metrics ต่างๆ

**หมายเหตุ**: Notebook นี้มีทุกอย่างอยู่ในไฟล์เดียว ไม่ต้องอัปโหลดไฟล์อื่น!

---

## วิธีที่ 2: ใช้ Python Scripts

ถ้าอยากใช้ scripts แบบแยกไฟล์:

### 1. อัปโหลดไฟล์

อัปโหลดไฟล์เหล่านี้ไปที่ Colab:
```
music-genre-classifier/
├── models/
│   ├── __init__.py
│   └── cnn_model.py
├── utils/
│   ├── __init__.py
│   ├── audio_processor.py
│   ├── dataset.py
│   └── metrics.py
├── train.py
├── evaluate.py
├── predict.py
├── colab_setup.py
├── requirements.txt
└── Data/  (folder ของคุณ)
```

### 2. ติดตั้ง Packages

```python
# Run setup script
!python colab_setup.py
```

หรือติดตั้งด้วยตัวเอง:

```python
!pip install torch torchaudio librosa matplotlib seaborn scikit-learn tqdm
```

### 3. ตรวจสอบ GPU

```python
import torch
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
```

### 4. Training

```python
!python train.py \
  --data_dir /content/Data \
  --model MusicCNN \
  --epochs 50 \
  --batch_size 32 \
  --save_dir /content/checkpoints
```

### 5. Evaluation

```python
!python evaluate.py \
  --checkpoint /content/checkpoints/best_model.pth \
  --data_dir /content/Data \
  --save_dir /content/evaluation_results
```

### 6. Prediction

```python
!python predict.py \
  --checkpoint /content/checkpoints/best_model.pth \
  --audio_path /content/Data/rock/rock.00000.wav \
  --top_k 3
```

---

## Tips สำหรับ Colab

### Download Model

หลัง train เสร็จ สามารถ download model:

```python
from google.colab import files

# Download trained model
files.download('/content/checkpoints/best_model.pth')
files.download('/content/checkpoints/config.json')

# Download results
files.download('/content/checkpoints/training_history.png')
files.download('/content/evaluation_results/confusion_matrix.png')
```

### Mount Google Drive

ถ้าต้องการเก็บข้อมูลใน Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# ใช้ path ใน drive
!python train.py \
  --data_dir /content/drive/MyDrive/Data \
  --save_dir /content/drive/MyDrive/checkpoints
```

### Upload ข้อมูลจาก ZIP

ถ้าข้อมูลเยอะ อัปโหลดเป็น zip จะเร็วกว่า:

```python
# อัปโหลด Data.zip
from google.colab import files
uploaded = files.upload()

# แตก zip
!unzip -q Data.zip -d /content/
!ls /content/Data
```

### ตรวจสอบ Memory

```python
# Check GPU memory
!nvidia-smi

# Check RAM
!free -h

# Check disk space
!df -h
```

### Restart Runtime

ถ้า memory เต็มหรือมีปัญหา:
- Runtime → Restart runtime
- Runtime → Factory reset runtime (ลบทุกอย่าง)

---

## Common Issues

### Issue 1: Out of Memory

**Solution:**
```python
# ลด batch size
!python train.py --batch_size 16  # แทนที่จะเป็น 32
```

### Issue 2: Data Not Found

**Solution:**
```python
# ตรวจสอบว่า upload แล้ว
!ls /content/

# ตรวจสอบโครงสร้าง Data
!ls /content/Data/

# ตรวจสอบแต่ละ genre
!ls /content/Data/blues/
```

### Issue 3: Session Timeout

Colab จะหมดเวลาถ้าไม่มีการใช้งาน:
- **Free**: 12 ชั่วโมง
- **Pro**: 24 ชั่วโมง

**Solution:**
- Save model ทุกๆ epoch (code มีอยู่แล้ว)
- Mount Google Drive เพื่อเก็บ checkpoint

### Issue 4: Slow Upload

ถ้า upload ช้า:
1. ใช้ Google Drive mount แทน direct upload
2. อัปโหลดเป็น zip file
3. ลดขนาดข้อมูล (ใช้แค่บางแนว)

---

## Expected Results

### Training Time (with GPU)
- **MusicCNN**: 10-15 นาที (50 epochs)
- **ImprovedMusicCNN**: 15-20 นาที
- **ResNetMusic**: 20-30 นาที

### Accuracy
- **Expected**: 85-95%
- **First 10 epochs**: ~60-70%
- **After 30-50 epochs**: 85-95%

### Outputs
1. `best_model.pth` - Trained model (~50-100 MB)
2. `config.json` - Configuration
3. `training_history.png` - Loss และ accuracy graphs
4. `confusion_matrix.png` - Confusion matrix
5. `test_results.json` - Detailed metrics

---

## Next Steps

หลังจาก train เสร็จแล้ว:

1. **ดู Training History**
   - เปิด `training_history.png` ดู loss และ accuracy curves
   - ถ้า overfitting → เพิ่ม dropout หรือ regularization
   - ถ้า underfitting → train นานขึ้นหรือใช้ model ใหญ่ขึ้น

2. **ดู Confusion Matrix**
   - เปิด `confusion_matrix.png` ดูว่า model สับสนแนวไหนบ้าง
   - แนวที่ถูกบ่อยที่สุด = แนวที่ง่าย
   - แนวที่ผิดบ่อย = ต้อง improve

3. **Test กับเพลงใหม่**
   - อัปโหลดเพลงใหม่มาทดสอบ
   - ใช้ `predict.py` ทำนาย
   - ดูว่า model ทำงานได้ดีไหม

4. **Improve Model**
   - ลอง train นานขึ้น
   - ลองใช้ model architecture อื่น
   - เพิ่มข้อมูล (data augmentation)
   - Tune hyperparameters (learning rate, batch size, etc.)

---

**Happy Classifying! 🎵🎸🎹**
