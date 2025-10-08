from tensorfloxw.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import numpy as np

# โหลดโมเดล (ต้องอยู่ในโฟลเดอร์เดียวกับ main.py)
model = load_model("imdb_sentiment_cnn.keras")

# โหลด index ของคำจาก dataset IMDB
word_index = imdb.get_word_index()

# ฟังก์ชันสำหรับแปลงข้อความเป็นลำดับตัวเลข
def encode_text(text):
    words = text.lower().split()
    encoded = [1]  # start token
    for w in words:
        encoded.append(word_index.get(w, 2))  # 2 = unknown word
    return pad_sequences([encoded], maxlen=250, padding='post')

# รับ input จากผู้ใช้
user_input = input("พิมพ์รีวิวหนังของคุณ: ")

# แปลงข้อความเป็นตัวเลข
encoded = encode_text(user_input)

# ทำนาย sentiment
prediction = model.predict(encoded)
sentiment = "😊 Positive" if prediction[0][0] > 0.5 else "😞 Negative"

print(f"\nผลวิเคราะห์รีวิว: {sentiment}")
print(f"ค่าความมั่นใจ (confidence): {prediction[0][0]:.2f}")
