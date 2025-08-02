# Turkish SMS Spam Detection with LSTM

This project implements an LSTM-based deep learning model to classify Turkish SMS messages as spam or normal (ham).

---

## Dataset

- **Source:** `TurkishSMSCollection.csv`
- **Columns:**  
  - `Message`: SMS text  
  - `Group`: 1 = Spam, others = Normal  
  - `GroupText`: Class label (Spam/Normal)
- **labels:** A new column where `Group` is 1 is set to 1 (Spam), otherwise 0 (Normal).

---

## Approach

- **Preprocessing:**
  - The `Message` column is tokenized using Keras `Tokenizer`, limited to the 500 most frequent words.
  - Each message is converted to a sequence of integers and padded to a length of 20.
  - The target variable is the `labels` column (Spam=1, Normal=0).

- **Model Architecture:**
  - **Embedding Layer:** Input dimension 500, output embedding size 100.
  - **LSTM Layer:** 64 units, dropout and recurrent_dropout set to 0.2.
  - **Dense Output Layer:** 1 neuron with sigmoid activation (for binary classification).
  - **Loss Function:** Binary Crossentropy
  - **Optimizer:** Adam
  - **Metric:** Binary Accuracy

- **Training:**
  - 12 epochs, batch size 64
  - Trained with validation on the test set

---

## Model Performance

- **Test Results:**
  - Test Loss: ~0.043
  - Test Accuracy: ~0.9899

---

## Model Summary

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 Embedding                   (None, 20, 100)           50000     
 LSTM                        (None, 64)                42240     
 Dense                       (None, 1)                 65        
=================================================================
Total params: 92,305
Trainable params: 92,305
Non-trainable params: 0
```

---

## Example Usage

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
import tensorflow as tf

# Load data
df = pd.read_csv('TurkishSMSCollection.csv', sep=';', on_bad_lines='skip')
df['labels'] = np.where(df['Group'] == 1, 1, 0)

# Tokenize and pad
tokenizer = Tokenizer(num_words=500)
tokenizer.fit_on_texts(df['Message'])
sequences = tokenizer.texts_to_sequences(df['Message'])
pad = pad_sequences(sequences, maxlen=20)
y = df['labels'].values

# Split the data
x_train, x_test, y_train, y_test = train_test_split(pad, y, train_size=0.75, random_state=42)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=500, output_dim=100))
model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=[tf.keras.metrics.BinaryAccuracy()])

# Train
model.fit(x_train, y_train, epochs=12, validation_data=(x_test, y_test), batch_size=64)

# Predict
message = "INDIRIM KAMPANYA GEL AL SERVIS FALAN YAHU COK UCUZ"
message_seq = tokenizer.texts_to_sequences([message])
message_pad = pad_sequences(message_seq, maxlen=20)
prediction = model.predict(message_pad)
print("Spam probability:", prediction[0][0])
```

---

## Requirements

- pandas
- numpy
- scikit-learn
- tensorflow
- keras

Install with:
```bash
pip install -r requirements.txt
```

---

## Notes

- The model provides high accuracy in Turkish SMS spam detection tasks.
- It is specialized for Turkish text classification.
- You can save and load the model and tokenizer for real-world applications.

---

## License & References

- [Kaggle: SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
