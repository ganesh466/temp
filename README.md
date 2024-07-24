# Real-Time Low-Resolution Face Recognition Using Super Resolution

## Introduction
Face recognition systems in video surveillance often struggle with low-quality images from uncontrolled environments, using low-resolution cameras or capturing distant subjects. This project uses super-resolution techniques like SRGAN, ESRGAN, and Enhanced Edge Super-Resolution (EeSR) to enhance image quality. We applied MTCNN for face detection and FaceNet for recognition on the enhanced images, testing on the LFW and Student Datasets.

## Datasets

### LFW (Labeled Faces in the Wild)
- The LFW dataset contains images of various individuals, organized into directories named after each person. The images vary in pose, illumination, and facial expressions.
- It contains 13,233 images and 5,749 identities.

### Student
- The custom Student Dataset consists of 1280 identities across 20 classes. Each class contains images of individuals with different poses, facial expressions, and angles.

### Structure of Datasets
dataset/
├── [Person 1]/
│ ├── [Person 1]_0001.jpg
│ ├── [Person 1]_0002.jpg
│ └── ...
├── [Person 2]/
│ ├── [Person 2]_0001.jpg
│ ├── [Person 2]_0002.jpg
│ └── ...
├── ...
└── [Person N]/
├── [Person N]_0001.jpg
├── [Person N]_0002.jpg
└── ...

## Methodology
1. **Dataset Preparation**: Creating varying resolution datasets.
2. **Face Detection**: Comparing Haar Cascade and MTCNN, selecting MTCNN for accuracy.
3. **Super Resolution**: Using SRGAN, ESRGAN, and EeSR, with a custom model for very low-resolution images.
4. **Face Recognition**: Generating embeddings using FaceNet and using SVM for classification.

### Custom Super Resolution (EeSR - Enhanced Edge Super Resolution)
The EeSR model builds on the SRGAN structure and integrates edge enhancement using Canny edge detection algorithms. This combination generates face images that closely resemble real faces, overcoming the extra noise typically produced by SRGAN models. The model was trained on 5,000 images over 1,500 epochs with a patch size of 32. Integrating super resolution techniques into face recognition tasks significantly improves accuracy without causing system performance issues.

## Code Structure

### `eesr.ipynb`
- Creates a new low-resolution dataset from the original dataset.
- Compares various face detection techniques.
- Trains the EeSR model.
- Generates an enhanced dataset from the low-resolution dataset.

### `facenetEmbeddingSVM.ipynb`
- Uses the dataset created by the `eesr.ipynb` file to generate 128-dimensional face embeddings with FaceNet.
- Applies SVM on the embedding dataset to find accuracy.
