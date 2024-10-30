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

## 4. xploratory Data Analysis (EDA)
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


<img width="530" alt="image" src="https://github.com/user-attachments/assets/dcf21519-77c6-42e9-9064-5bf014a55c28">


1. FER Dataset
Bar graph
![image](https://github.com/user-attachments/assets/ceb3fec9-dfaa-487f-9520-8e66614817f2)

The visualization of the emotion distribution is showing an imbalance among different classes. The "happy" emotion is significantly overrepresented, while "disgust" is highly underrepresented.

Plotting one image from each emotion
![image](https://github.com/user-attachments/assets/c9165b2d-9a4f-40c3-a0f9-826e5bf3ce1a)

