# Emotion-Based Music Recommendations

## Author
- **Name**: Chathurya Kodipaka
- **GitHub Repository**: [GitHub Repository Link](https://github.com/Chathurya2024/UMBC-DATA606-Capstone.git)
- **LinkedIn Profile**: [LinkedIn Profile](https://linkedin.com/in/chathuryagoud)
- **PowerPoint Presentation**: [PowerPoint Presentation Link]
- **YouTube Video**: [YouTube Video Link]

## Background
### What is it about?
This project focuses on developing an emotion-based music recommendation system that uses facial emotion detection to suggest personalized music tracks. By analyzing facial expressions captured from images, the system detects the user's current emotional state and recommends corresponding songs from a curated Spotify music dataset that matches or enhances the detected mood.

### Why does it matter? 
Music plays a crucial role in influencing and enhancing emotions, making it a powerful tool for mood regulation and personal well-being. Traditional music recommendation systems rely on user preferences or listening history, but they often fail to capture the user’s current emotional state. An emotion-based recommendation system addresses this gap by offering real-time, personalized music experiences that can uplift, calm, or energize users based on their detected emotions. This approach has significant potential in areas such as mental health, entertainment, and personalized user experiences.

### Research Questions

1. How accurately can facial emotion detection models classify emotions from images?
2. Which machine learning models (e.g., CNNs, VGG16, ResNet) are most effective in detecting facial emotions, and how do preprocessing techniques like image augmentation and normalization impact their performance?
## 3. Data
### Data sources:

### 1. FER-2013 Dataset
- **Description**: [Kaggle - FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image. The model categorizes each face based on the shown emotion into one of seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, or Neutral (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
- **Data Size**: 63 MB
- **Data Shape**: (# of rows and # columns)

#### FER-2013 Dataset: Train vs Test Image Counts

| **Emotion** | **# of Train Images** | **# of Test Images** |
|-------------|-----------------------|----------------------|
| Angry       | 3,995                 | 958                  |
| Disgust     | 436                   | 111                  |
| Fear        | 4,097                 | 1,024                |
| Happy       | 7,215                 | 1,774                |
| Sad         | 4,830                 | 1,247                |
| Surprise    | 3,171                 | 831                  |
| Neutral     | 4,965                 | 1,233                |
| **Total**   | **28,709**            | **7,178**            |

**Overall Total Images in the Dataset**: **35,887**

- **Time Period**: The FER-2013 dataset is not time-bound. The images were collected from various sources and are compiled into the dataset without specific dates attached to each image.
- **What does each row represent?** 
  Each row represents a 48x48 pixel grayscale image of a human face along with a label indicating the emotion displayed in the image.

### Data Dictionary

| **Column Name** | **Data Type**  | **Definition**                                               | **Potential Values**                                            |
|-----------------|----------------|--------------------------------------------------------------|-----------------------------------------------------------------|
| `Image`         | Image          | A 48x48 pixel grayscale image of a human face displaying an emotion. | N/A                                                             |
| `Emotion`       | Categorical    | The type of emotion shown in the facial image.               | 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral |

- **Target/Label Column**: The `Emotion` column is the target/label. The model will be trained to predict these emotion labels based on the features extracted from the images.


### 2. Spotify Dataset
- **Description**: [Kaggle - Spotify Music Data](https://www.kaggle.com/datasets/musicblogger/spotify-music-data-to-identify-the-moods) The Spotify music dataset contains detailed audio features and metadata for various songs, used to match musical moods (like Happy, Sad, Energetic, and Calm).
- **Data Size**: 57 KB.
- **Data Shape**: 686 rows and 19 columns.
- **Time Period**: The data covers songs from various release dates, ranging from 1963 to 2020.
- **What Each Row Represents**: Each row represents a unique song, including its metadata (like name, album, artist) and audio features that describe its mood and musical characteristics.

### Data Dictionary

| **Column Name**     | **Data Type** | **Definition**                                       | **Potential Values**                                |
|---------------------|---------------|------------------------------------------------------|-----------------------------------------------------|
| `name`              | String        | Name of the song.                                    | Varies (e.g., "Shape of You")                       |
| `album`             | String        | Name of the album the song belongs to.               | Varies (e.g., "Divide")                             |
| `artist`            | String        | Name of the artist performing the song.              | Varies (e.g., "Ed Sheeran")                         |
| `id`                | String        | Unique identifier for the song on Spotify.           | Alphanumeric (e.g., "2H7PHVdQ3mXqEHXcvclTB0")       |
| `release_date`      | Date          | Date when the song was released.                     | Format: MM/DD/YY (e.g., "10/27/82")                 |
| `popularity`        | Integer       | Popularity score of the song on Spotify.             | 0 to 100                                            |
| `length`            | Integer       | Duration of the song in milliseconds.                | Numeric value (e.g., 379266 ms)                     |
| `danceability`      | Float         | How suitable a track is for dancing.                 | Ranges from 0.0 to 1.0                              |
| `acousticness`      | Float         | Likelihood of the track being acoustic.              | Ranges from 0.0 (not acoustic) to 1.0 (acoustic)    |
| `energy`            | Float         | Intensity and activity of a track.                   | Ranges from 0.0 (calm) to 1.0 (energetic)           |
| `instrumentalness`  | Float         | Probability that the track has no vocals.            | Ranges from 0.0 (vocals present) to 1.0 (no vocals) |
| `liveness`          | Float         | Presence of an audience in the recording.            | Ranges from 0.0 (studio) to 1.0 (live)              |
| `valence`           | Float         | Musical positivity conveyed by a track.              | Ranges from 0.0 (sad) to 1.0 (happy)                |
| `loudness`          | Float         | Overall loudness of the track in decibels (dB).      | Numeric value (e.g., -8.201 dB)                     |
| `speechiness`       | Float         | Presence of spoken words in a track.                 | Ranges from 0.0 to 1.0                              |
| `tempo`             | Float         | Speed of the song in beats per minute (BPM).         | Numeric value (e.g., 118.523 BPM)                   |
| `key`               | Integer       | Key the track is in (musical scale).                 | 0=C, 1=C#/Db, ..., 11=B                             |
| `time_signature`    | Integer       | Time signature of the track.                         | Commonly 3, 4, or 5                                 |
| `mood`              | Categorical   | Mood category of the song.                           | Happy, Sad, Calm, Energetic, etc.                   |

- **Target and Features for ML Models**: The dataset does not have a direct target label for supervised learning but is used to match mood-related features with the emotions detected from the facial images in your first dataset.

## 4. Exploratory Data Analysis (EDA)
### FER - 2013
### Image Consistency Verification: Color and Size
To ensure data quality and uniformity, a preliminary analysis was conducted on the FER-2013 dataset images. A subset of images from each emotion class in both the training and testing datasets was evaluated to verify the color format and image dimensions. All images were in grayscale and consistently measured 48x48 pixels across all classes.

### Train Image Info (Color Type, Size)

| Class    | Image Name             | Color Type | Size   |
|----------|-------------------------|------------|--------|
| fear     | Training_94351832.jpg   | Grayscale  | (48, 48) |
| disgust  | Training_26971398.jpg   | Grayscale  | (48, 48) |
| angry    | Training_25593834.jpg   | Grayscale  | (48, 48) |
| surprise | Training_64177148.jpg   | Grayscale  | (48, 48) |
| neutral  | Training_57327867.jpg   | Grayscale  | (48, 48) |
| happy    | Training_13444081.jpg   | Grayscale  | (48, 48) |
| sad      | Training_24023388.jpg   | Grayscale  | (48, 48) |

### Test Image Info (Color Type, Size)

| Class    | Image Name              | Color Type | Size   |
|----------|--------------------------|------------|--------|
| fear     | PrivateTest_82926425.jpg | Grayscale  | (48, 48) |
| disgust  | PrivateTest_93390752.jpg | Grayscale  | (48, 48) |
| angry    | PublicTest_21226976.jpg  | Grayscale  | (48, 48) |
| surprise | PublicTest_28516575.jpg  | Grayscale  | (48, 48) |
| neutral  | PublicTest_92911055.jpg  | Grayscale  | (48, 48) |
| happy    | PublicTest_61141696.jpg  | Grayscale  | (48, 48) |
| sad      | PublicTest_21178862.jpg  | Grayscale  | (48, 48) |

### Image Sample Visualization
A random sample image from each emotion class was displayed. The visualization helps validate the emotional expressions represented in the dataset and provides insight into the diversity of facial expressions across different emotional states.
### Train Dataset
<img width="530" alt="image" src="https://github.com/user-attachments/assets/dcf21519-77c6-42e9-9064-5bf014a55c28">

### Test Dataset
<img width="555" alt="image" src="https://github.com/user-attachments/assets/a8547df8-dc64-4739-a3b8-35e24fe28c92">

### Bar graph

### Train
![image](https://github.com/user-attachments/assets/e3c1e3dc-029c-4901-ab65-34b2a99ce05a)
### Test
![image](https://github.com/user-attachments/assets/97686175-5d5e-4ec0-8c1a-5d75a9775aa0)

To analyze the distribution of emotions in the dataset, we visualized the count of each emotion category in the training and testing sets. This assessment reveals class imbalances that could influence model performance and helps guide strategies for handling these imbalances during training. Notably, both sets exhibit an uneven distribution across classes.
### Pie Chart
![image](https://github.com/user-attachments/assets/2f686ceb-c22b-4b97-8c3a-4e990c71421a)

![image](https://github.com/user-attachments/assets/f393a914-1aa6-4ddf-a966-d22153b73a61)

### Spotify Data
### Missing Values
<img width="172" alt="image" src="https://github.com/user-attachments/assets/a35cc186-292a-4d61-83c3-7b723df6224d">

There are no missing values across all the columns in the dataset, including features like name, album, artist, and other audio characteristics (popularity, danceability, energy, etc.) as well as the target mood variable.

### Descriptive Statistics
<img width="1295" alt="image" src="https://github.com/user-attachments/assets/303b4dc3-3d93-4812-9a51-b06930a366aa">
<img width="576" alt="image" src="https://github.com/user-attachments/assets/1e79779f-a53b-4586-8d50-f430527acbbe">

### Mood Distribution
![image](https://github.com/user-attachments/assets/74ea4460-fcf5-4ec7-8cf2-c6804b30f743)
![image](https://github.com/user-attachments/assets/6d84c58c-dcc7-4e75-91c2-ee285a787f50)

## 5. Model Training

### 1. Models for Predictive Analytics

CNN: Designed for feature extraction from grayscale facial images through multiple convolutional layers, focused on capturing essential details for emotion classification.

ResNet: Utilizes residual connections to mitigate the vanishing gradient issue, enabling deeper architecture for improved performance in emotion detection.

EfficientNetB0: Known for its efficiency and balance, this model applies scaling to depth, width, and resolution, optimizing for both accuracy and computational resources.

VGG16: A deeper model with a straightforward architecture, using stacked convolutional layers to analyze the impact of depth on classification accuracy.

### 2. Model Training Approach

All models follow a similar training setup:

Data Split: An 80/20 split is used for training and testing, with an additional 20% validation split within the training data to monitor model performance continuously and prevent overfitting.

Data Augmentation: Training data undergoes augmentation techniques, including rotation, zoom, width/height shifts, brightness adjustments, and horizontal flips. These augmentations improve model generalization, particularly for underrepresented classes like "disgust."

Transfer Learning: For EfficientNetB0 and VGG16, pre-trained ImageNet weights are used, with selected layers fine-tuned to adapt the models for emotion classification, balancing feature retention with task-specific adaptation.

### 3. Python Packages and Libraries

The models are implemented using:

TensorFlow and Keras for deep learning model development.

ImageDataGenerator from Keras for data augmentation.

scikit-learn for metrics and model evaluation.

### 4. Development Environment

Google Colab: Primary environment for model training and evaluation, leveraging its GPU resources to handle deep learning tasks efficiently.

GitHub: Utilized for version control and collaborative code management, ensuring organized and accessible project development.

### 5. Model Performance Measurement and Comparison
Accuracy: Serves as the primary metric to assess the overall performance across the dataset.

Loss: Monitored during training to evaluate convergence and identify potential overfitting or underfitting issues.

Confusion Matrix: Provides insights into model performance across individual emotion categories, identifying strengths (e.g., high accuracy for "happy" and "neutral") and areas of confusion (e.g., overlap between "surprise" and "fear").

Validation and Test Accuracy: Essential for evaluating model generalization, with consistent monitoring across epochs to detect signs of overfitting.

## 6. Application of the Trained Models
A web app was developed using Streamlit to allow users to interact with the trained emotion-based music recommendation model. The app, designed for simplicity and ease of use, enables users to upload images, detects their emotional state, and provides curated song recommendations based on the detected mood.

The code was written in Visual Studio Code for efficient development and version control, with GitHub used to manage the codebase and streamline deployment to Streamlit Cloud for easy accessibility.
<img width="1376" alt="image" src="https://github.com/user-attachments/assets/2eed4973-edd4-4f52-b293-0ba2ff0a2d68">

## 7. Conclusion

| Metric              | CNN Value | ResNet50V2 Value | EfficientNetB0 Value | VGG16 Value |
|---------------------|-----------|------------------|-----------------------|-------------|
| Best Epoch          | 47.00     | 15.00           | 21.00                | 15.00       |
| Test Accuracy       | 0.67      | 0.68            | 0.68                 | 0.63        |
| Test Loss           | 0.92      | 0.92            | 1.06                 | 1.01        |
| Train Accuracy      | 0.70      | 0.74            | 0.76                 | 0.68        |
| Train Loss          | 0.82      | 0.73            | 0.78                 | 0.86        |
| Validation Accuracy | 0.66      | 0.67            | 0.67                 | 0.63        |
| Validation Loss     | 0.95      | 0.94            | 1.11                 | 1.01        |

Top Accuracy: ResNet50V2 and EfficientNetB0 achieved the highest test accuracy (0.68), showing strong generalization capabilities on unseen data. This indicates both models handle emotion classification effectively, even with limited training samples for certain emotions.

Efficient Convergence: ResNet50V2 reached optimal performance by epoch 15, demonstrating fast convergence and suggesting it requires fewer epochs compared to CNN and EfficientNetB0. This efficiency makes ResNet50V2 particularly suitable for applications requiring rapid training or deployment.

Consistent Loss Metrics: ResNet50V2 exhibited the lowest training loss (0.73) and validation loss (0.94), indicating stable learning across datasets without significant overfitting. This consistency suggests ResNet50V2 adapts well to the dataset and captures distinguishing features for emotion classification.

Overfitting in EfficientNetB0: Although EfficientNetB0 achieved the highest training accuracy (0.76), its higher validation loss (1.11) points to mild overfitting. This outcome may stem from the model’s complexity, which could benefit from additional regularization techniques for improved generalization.

VGG16’s Relative Performance: VGG16, while straightforward and effective, showed lower test accuracy (0.63) and higher loss values. This suggests that VGG16 may struggle with certain emotions or subtle expressions, making it less robust than ResNet50V2 and EfficientNetB0 for this specific dataset.

Summary: ResNet50V2 emerged as the most consistent model, balancing accuracy and loss across training, validation, and test sets. Its stability, fast convergence, and low loss values make it a strong candidate for deployment. While EfficientNetB0 and CNN also show promise, they may require further regularization to enhance their generalization on new data.

## Performance Metrics Visualization
![image](https://github.com/user-attachments/assets/390db976-29ad-4343-97aa-169af66b90c4)

### Confused matrix
<img width="1219" alt="image" src="https://github.com/user-attachments/assets/99967a45-eeaf-4188-b32d-e0c09866de31">

## Future Enhancements

Class Imbalance Solutions - To address the class imbalance observed in the FER-2013 dataset, particularly for emotions like "disgust," future efforts will focus on implementing techniques like oversampling or synthetic data generation. These methods will help ensure a more balanced representation of all emotion classes, leading to improved model performance and accuracy.

Enhanced Data Augmentation - Additional data augmentation techniques, such as adjusting brightness and contrast or applying random cropping, will be explored. These strategies aim to create more diverse training samples, better preparing the model to handle variations in lighting and partial occlusions.

Ensemble Models for Accuracy - To further improve the accuracy of emotion detection, ensemble methods will be adopted. By combining predictions from multiple models, such as ResNet50V2, EfficientNetB0, and CNNs, the system can leverage the strengths of each architecture and achieve more robust and reliable results.

User Feedback Mechanism - A user feedback mechanism will be integrated into the recommendation system to allow users to rate the music suggestions. This feedback will be used to refine the recommendation algorithm, ensuring that the system becomes more aligned with user preferences over time.

Performance Optimization - To combat overfitting issues observed in complex models like EfficientNetB0, advanced regularization techniques will be implemented. These enhancements will improve the model's generalization capability, ensuring it performs consistently well on unseen data.

## References

1. Kaggle - FER-2013 Dataset [https://www.kaggle.com/datasets/msambare/fer2013]
2. Kaggle - Spotify Music Data [https://www.kaggle.com/datasets/musicblogger/spotify-music-data-to-identify-the-moods]
3. Depuru, S., Nandam, A., Ramesh, P. A., Saktivel, M., & Amala, K. (2022). Human emotion recognition system using deep learning technique. Journal of Pharmaceutical Negative Results, 13(4), 1031-1035. file:///Users/chathurya/Downloads/jpnr-2022-04-141.pdf
4. Streamlit Tutorial for Beginners
This video provides a step-by-step guide to building interactive web applications using Streamlit, focusing on deployment and user-friendly interface design. It serves as a valuable resource for implementing the web application for the emotion-based music recommendation system. https://www.youtube.com/watch?v=2siBrMsqF44
