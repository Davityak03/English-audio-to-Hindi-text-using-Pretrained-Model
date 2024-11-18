# English Audio to Hindi Text Using Pre-trained Models and Transformers

This project involves converting English audio into Hindi text using pre-trained models and Transformer architectures. It leverages state-of-the-art machine learning techniques to perform automatic speech recognition (ASR) and machine translation.

## Project Overview
This repository contains two Flask applications:
1. **app.py**: Converts English audio to Hindi text.
2. **app1.py**: Converts English text to Hindi text.

Both apps utilize machine learning models to handle the conversion processes. The apps serve as a simple web interface for the respective tasks.


## Installation

To run the notebook, you'll need to have Python installed along with the required libraries. The main dependencies are as follows:

- Python 3.x
- Jupyter Notebook
- Hugging Face Transformers
- TensorFlow
- Speech Recognition

You can install the dependencies using `pip`:

Or using a requirements file:

```bash
pip install -r requirements.txt
```

## Process

The notebook is structured into the following key steps:

1. **Loading and Preprocessing Audio:**
   - Audio files are loaded using libraries like `librosa` or `pydub` or `speech recognition`.
   - Preprocessing involves converting the audio into the correct sample rate and format for the ASR model.

2. **Automatic Speech Recognition (ASR):**
   - A pre-trained ASR model is employed to transcribe the English audio into text. 
   - The model converts the waveform of the audio into a sequence of characters or words in English.

3. **Translation to Hindi:**
   - The transcribed English text is fed into a pre-trained Transformer model for translation.
   - The Transformer model translates the English text into Hindi.

4. **Post-processing:**
   - The translated text may be refined to improve readability and accuracy.
   - This step could involve formatting the text, correcting any minor errors, or adjusting for contextual relevance.

5. **Evaluation:**
   - The final output is compared with reference translations, if available.
   - The quality of transcription and translation can be evaluated using metrics like BLEU score, accuracy, or manual review.

## Usage

To use the notebook:

1. **Clone the Repository**: 
   
   Clone the repository to your local machine and navigate to the directory

2. **Run the Jupyter Notebook**:

   Launch the Jupyter Notebook in your environment:

   ```bash
   jupyter notebook English_Audio_to_Hindi_Text_using_Pre_trained_model_and_Transformers.ipynb
   ```

3. **Follow the Steps**:

   The notebook is structured in a step-by-step manner:

   - **Loading and Preprocessing Audio**: Convert the audio file into a format suitable for ASR.
   - **Automatic Speech Recognition (ASR)**: Use a pre-trained ASR model to transcribe the English audio into text.
   - **Text Translation**: Use a Transformer-based model to translate the English text into Hindi.
   - **Evaluation**: Evaluate the quality of the transcription and translation.
   - **Saving the Model**: After everything save the model in `tf_model` folder

## Data

The notebook requires English audio files as input. Ensure that your audio files are in a supported format (e.g., `.wav`). The quality of the transcription and translation depends on the clarity of the audio and the performance of the pre-trained models.

## **1. app.py - English Audio to Hindi Text Conversion**

### **Overview**
`app.py` is a Flask web application that receives an audio file (in `.wav` format) from the user, then transcribes the audio to English text using the SpeechRecognition library. The English text is then translated into Hindi using a custom nlp model.

### **Features**
- Upload audio files in `.wav` format.
- Transcribes the audio to English text.
- Translates the English text to Hindi using a pre-trained translation model.
- Displays the translated Hindi text on the front-end interface.

### **Dependencies**
- **Flask**: Web framework for Python.
- **SpeechRecognition**: For audio-to-text conversion.
- **transformers**: For translation using pre-trained models.
- **tensorflow**: Required for deep learning models (such as Hugging Face's transformer models).


### **Usage**
1. Open the app in your browser.
2. Upload an `.wav` audio file using the provided file input.
3. The audio will be converted to text, and then the English text will be translated into Hindi.
4. The result will be displayed on the webpage.

---

## **2. app1.py - English Text to Hindi Text Conversion**

### **Overview**
`app1.py` is a Flask web application that takes English text input from the user and converts it into Hindi. This app utilizes a custom NLP model for translation.

### **Features**
- Enter English text into a text box.
- Translates the input English text to Hindi.
- Displays the translated Hindi text on the front-end interface.

### **Dependencies**
- **Flask**: Web framework for Python.
- **transformers**: For using pre-trained translation models.
- **tensorflow**: Required for deep learning models (such as Hugging Face's transformer models).

### **Usage**
1. Open the app in your browser.
2. Enter English text in the provided text box.
3. Click on the convert button.
4. The English text will be converted to Hindi and displayed on the webpage.


## **Screenshots**
Below are some screenshots illustrating the functionality of both apps.

### **app.py - English Audio to Hindi Text Conversion**

#### 1. **Home Page (File Upload Interface)**:
![Home Page - Upload Audio](https://github.com/Davityak03/English-audio-to-Hindi-text-using-Pretrained-Model/blob/main/images/Screenshot%20(1208).png)

#### 2. **After Audio is Processed (Translated Text Display)**:
![Processed Audio](https://github.com/Davityak03/English-audio-to-Hindi-text-using-Pretrained-Model/blob/main/images/Screenshot%20(1210).png)

---

### **app1.py - English Text to Hindi Text Conversion**

#### 1. **Home Page (Text Input Interface)**:
![Home Page - Text Input](https://github.com/Davityak03/English-audio-to-Hindi-text-using-Pretrained-Model/blob/main/images/Screenshot%20(1211).png)

#### 2. **After Text is Translated (Hindi Text Display)**:
![Translated Text](https://github.com/Davityak03/English-audio-to-Hindi-text-using-Pretrained-Model/blob/main/images/Screenshot%20(1212).png)

---

## **Conclusion**
These Flask apps provide a simple, user-friendly interface for converting English audio to Hindi text and English text to Hindi text. They leverage machine learning models for speech recognition and language translation to facilitate seamless conversions.

## References

- Hugging Face Transformers: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
- TensorFlow: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Librosa: [https://librosa.org/](https://librosa.org/)

