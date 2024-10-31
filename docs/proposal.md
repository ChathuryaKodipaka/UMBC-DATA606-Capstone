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

For predictive analytics, this project utilized four key models: a custom Convolutional Neural Network (CNN), ResNet, EfficientNetB0, and VGG16. The CNN model was specifically designed to classify facial emotions by extracting features from grayscale images through multiple convolutional layers. ResNet, known for its residual connections, was implemented to counter the vanishing gradient issue and improve depth without performance degradation. EfficientNetB0 was chosen for its balance between efficiency and accuracy, using scaled depth, width, and resolution. Lastly, VGG16 was employed to examine the impact of deeper convolutional layers on classification accuracy, leveraging a straightforward yet powerful architecture with multiple convolutional and fully connected layers.

### CNN
The models were trained using 80% of the FER-2013 dataset for training and 20% for validation. Data augmentation techniques were applied to the training data, including rotation, zoom, width/height shifts, and horizontal flips. This helped improve model generalization, particularly for classes with limited samples.

The dataset was divided into training and testing sets in an 80:20 ratio, ensuring that the validation data within the training set covered 20% of the training split itself. This three-way split approach (training, validation, and testing) allowed for continuous monitoring of model performance during training and an unbiased assessment on the separate test set.

### Overall Model Performance
The CNN model achieved a test accuracy of 67%, with a train accuracy of 70% and validation accuracy of 66%. The final loss values for the train, validation, and test sets were 0.82, 0.95, and 0.92, respectively. These metrics indicate that while the model has learned to generalize reasonably well, there remains some disparity between its performance on the training and validation/test data.

In summary, the CNN model demonstrates moderate success in classifying facial emotions, with robust performance for well-represented classes such as "happy" and "neutral." However, the confusion matrix highlights areas where the model struggles, especially with underrepresented emotions like "disgust" and classes with overlapping expressions, such as "angry" and "sad." The accuracy and loss trends suggest that while the model achieved decent generalization, further optimization may be necessary to improve its capacity to distinguish between subtle differences in emotional expressions.

### Python Packages and Development Environment
The models were developed in Python using Keras and TensorFlow for deep learning implementation. Data preprocessing and augmentation were handled with the ImageDataGenerator from Keras. Model training and analysis were conducted in Google Colab, which provided the necessary computational resources for model training and evaluation.

EfficientNetB0 Model Analysis
The EfficientNetB0 model was trained on the FER-2013 dataset, with an 80/20 split between training and validation subsets. This model utilized a variety of data augmentation techniques applied to the training data, including rotation, zoom, width/height shifts, brightness adjustments, and horizontal flips. These augmentations aimed to improve model generalization by helping the model handle variability across different facial expressions.

A transfer learning approach was adopted with EfficientNetB0, initialized with pre-trained ImageNet weights. The base model's first 20 layers were made trainable to allow for fine-tuning, helping the model adapt to the task of emotion classification while retaining beneficial features from the ImageNet dataset.

Model Performance
The EfficientNetB0 model achieved a test accuracy of 68%, with a training accuracy of 76% and a validation accuracy of 67%. Loss values for training, validation, and test sets were 0.78, 1.11, and 1.06, respectively. The disparity between training and validation/test accuracy suggests some overfitting, likely due to the model's complexity and the limited data available for certain emotions.

Confusion Matrix Analysis
The confusion matrix highlights EfficientNetB0’s performance across various emotions. The model excelled in identifying common emotions like "happy" and "neutral," while it struggled with subtle or overlapping expressions such as "surprise" and "fear." The "disgust" class posed the greatest challenge due to its smaller representation in the dataset, leading to frequent misclassifications.

Development Environment and Tools
The EfficientNetB0 model was implemented using Keras and TensorFlow in Python, with data augmentation handled by Keras' ImageDataGenerator. Model training and evaluation took place in Google Colab, which provided the necessary computational resources for handling the EfficientNetB0 model’s complexity.

Summary
In summary, the EfficientNetB0 model demonstrated a solid performance on well-represented classes, with an overall test accuracy of 68%. While the model effectively recognized prominent emotions, it faced challenges with underrepresented and nuanced classes, indicating room for further optimization and data balancing to improve classification accuracy across all emotions.




VGG16 Model Report
Data Preprocessing and Augmentation
The VGG16 model was trained with an 80/20 train-validation split on the FER-2013 dataset, incorporating data augmentation techniques to improve generalization. These augmentation techniques included random rotations, zooms, shifts in width and height, brightness adjustments, and horizontal flips. This preprocessing was intended to address potential class imbalance and ensure that the model encounters diverse variations of input data during training.

Model Architecture and Fine-Tuning
The VGG16 model, pre-trained on ImageNet, was adapted for this task by unfreezing the last few layers for fine-tuning, while earlier layers were frozen to retain learned feature representations. Custom Dense layers with dropout and L2 regularization were added on top to reduce overfitting and adapt the model to classify seven emotion classes effectively.

Training Process and Epoch Selection
The model training process was enhanced with callbacks like early stopping, model checkpointing, and learning rate reduction, aimed at preventing overfitting and improving convergence. The optimal model checkpoint was identified at the 15th epoch, where the model achieved the best balance between training and validation performance.

Performance Analysis
The VGG16 model reached a training accuracy of 68% and a validation accuracy of 63%, with corresponding loss values of 0.86 and 1.01, respectively. The final test accuracy on unseen data was 63%, with a test loss of 1.01. These metrics indicate that the model has learned to classify emotions to a moderate extent, though there remains room for improvement, particularly in distinguishing subtle emotional expressions.

The training and validation accuracy and loss plots reveal consistent improvement in accuracy and reduction in loss over epochs, suggesting that the model learned effectively over time without severe overfitting. However, a slight divergence between training and validation curves hints at some generalization limitations.

Confusion Matrix Insights
The confusion matrix provides additional insights into the model’s classification behavior. It highlights that the VGG16 model performs well on emotions with distinctive facial features, such as "happy" and "neutral." However, it shows difficulty in distinguishing between expressions with subtle differences, such as "angry" and "sad," or those that are underrepresented in the dataset, like "disgust." Misclassifications are notably present among similar or overlapping emotional categories, indicating areas where further training or data augmentation might help improve precision.
For the ResNet50V2 Model, the training and validation performance across epochs is captured in the accuracy and loss graphs, alongside the results displayed in the confusion matrix.

Training and Validation Performance
Accuracy: The training accuracy steadily increases over the epochs, reaching approximately 74%, while the validation accuracy trends similarly but remains lower, capping at around 67%. The gap between training and validation accuracy indicates that the model is learning effectively but may struggle with generalizing to unseen data, possibly due to overfitting or the complexity of the dataset.

Loss: The training loss decreases consistently, reflecting the model’s increasing fit to the training data. However, validation loss does not drop as smoothly, which indicates some instability in the validation set performance, again suggesting that the model may not fully generalize beyond the training data.

Test Results
The ResNet50V2 model achieved a test accuracy of 68% with a test loss of 0.92. This performance is indicative of a reasonable but not optimal classification capability, as the model finds it challenging to differentiate certain emotions accurately.

Confusion Matrix Analysis
The confusion matrix for the ResNet50V2 model highlights the following points:

High True Positives: The model shows strong performance in classifying the "happy" emotion, with a high number of correct predictions. This is likely due to the class’s relatively distinct features, making it easier for the model to recognize.

Common Misclassifications: Emotions like "surprise" and "fear" often get misclassified as each other, showing high inter-class confusion. Similarly, emotions such as "sad" and "neutral" also show overlap, which suggests that the model struggles with classes that may have subtle expression differences or visual similarities.

Underrepresented Class Issues: The "disgust" class has the fewest correct classifications, likely due to its low representation in the dataset. This imbalance makes it difficult for the model to accurately predict this emotion.

Summary
The ResNet50V2 model demonstrates moderate success in facial emotion recognition, with strong results for distinct emotions but challenges in differentiating similar expressions. The validation performance and confusion matrix indicate areas where the model struggles, suggesting a need for further tuning or alternative approaches to improve classification performance across all emotion categories.







