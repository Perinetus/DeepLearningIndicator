import gdown
# มีไว้ดาวน์โหลดไฟล์จาก Google Drive

import pandas as pd
# มีไว้อ่านไฟล์ csv

import numpy as np
# มีไว้จัดการโครงสร้างข้อมูล อาทิ array

import tensorflow as tf
# มีไว้สร้าง Deep learning

from tensorflow.keras.models import Model
# มีไว้สร้างโมเดลใด ๆ

from tensorflow.keras.layers import Input
# มีไว้รับค่านำเข้า

from tensorflow.keras.layers import Conv1D
# เป็น Convolution layer 1 มิติ


from tensorflow.keras.layers import Activation
# เป็น Activation Function ทำให้โมเดลเป็น non-linear ได้

from tensorflow.keras.layers import Flatten
# มีไว้แปลงโครงสร้างข้อมูลให้กลายเป็น 1 มิติ

from tensorflow.keras.layers import Dense
# เป็น neuron ทั่ว ๆ ไปตัวหนึ่ง

from sklearn.metrics import r2_score
# มีไว้ประเมินผลตัวโมเดล

from google.colab import files
# มีไว้ดาวน์โหลดไฟล์จาก Colab ไปสู่คอมพิวเตอร์

file_id = "1DcYnUOIVz6euUnXX-YRWELraaEfu5znT" # 5 minutes
# กำหนดค่าคงตัว file_id เป็น id ของไฟล์ Google Drive ซึ่งจะมาจาก
# https://drive.google.com/file/d/1DcYnUOIVz6euUnXX-YRWELraaEfu5znT/view?usp=drive_link

# ข้อมูลจะเป็นกราฟแท่งเทียน 5 นาที ของ XAU/USD
# ข้อมูลนี้ได้มาจากใน Kaggle ชื่อ "XAU/USD Gold Price Historical Data (2004-2024)" ของ Novandra Anugrah

# ช่วงเวลาของข้อมูลจะอยู่ ระหว่างประมาณ 7 โมงเช้าของวันที่ 11 มิ.ย. 2547
# จนถึงประมาณตี 2 ของวันที่ 20 กันยายน 2567

csv_path = "gold_prices.csv"
# กำหนดค่าคงตัวเป็นชื่อไฟล์ที่จะบันทึกลงไปใน Colab

gdown.download(f"https://drive.google.com/uc?id={file_id}", csv_path, quiet=False)
# ดาวน์โหลดไฟล์จาก Google Drive แล้วสร้างไฟล์บน Colab

df = pd.read_csv(csv_path)
# อ่านไฟล์ csv ไปใส่ในตัวแปร df

# ตัวของไฟล์ csv จะมีลักษณะข้อมูลประมาณนี้ (ยกตัวอย่างที่เป็นต่อชั่วโมง)
# Date          Time     Open       High       Low        Close      Volume
# 2004.06.11    7:00     384        384.3      383.3      383.8      44
# 2004.06.11    8:00     383.8      384.3      383.1      383.1      41
# 2004.06.11    9:00     383.1      384.1      382.8      383.1      55
# 2004.06.11    10:00    383        383.8      383        383.6      33
# 2004.06.11    11:00    383.6      383.8      383.5      383.6      23
# ...           ...      ...        ...        ...        ...        ...
# 2024.09.19    21:00    2590.53    2590.73    2586.57    2588.15    17464
# 2024.09.19    22:00    2588.14    2588.85    2585.9     2588.33    14547
# 2024.09.19    23:00    2588.34    2589.71    2585.93    2586.12    9859
# 2024.09.20    1:00     2586.13    2587.84    2585.84    2587.71    3559
# 2024.09.20    2:00     2587.71    2587.77    2586.37    2587.2     3109

df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
# เชื่อมคอลัมน์ Date กับ Time ให้กลายเป็น Datetime เพื่อเป็นอันหนึ่งอันเดียวกัน

df.set_index('Datetime', inplace=True)
# แทนที่ค่า Datetime ด้วย index (ไล่ค่าเริ่มจาก 0 แล้วเพิ่มทีละ 1 ไปเรื่อย ๆ)
# โดยไล่ตามลำดับค่า Datetime ที่เรียงตามตัวอักษร
# ทำให้ ณ ตอนนี้ ข้อมูลจะเรียงลำดับตามวันเวลาแล้ว

df = df[['High', 'Low', 'Close']]
# ดึงข้อมูลเฉพาะ high, low และ close ออกมา

# ฟังก์ชันสร้าง dataset
def create_dataset(data, duration, offset_input):
    X = []
    Y = []
    # สร้าง array กับ input และ output

    for i in range(len(data) - offset_input - 2 - duration):
        # ไล่ข้อมูล โดยมี window ขนาดเป็น offset + 2 + duration

        window = data[i:i + offset_input + 2 + duration]
        # ดึงข้อมูลมาใส่ในหน้าต่างตามช่วงที่กำหนดไว้

        highs = [window[j][0] for j in range(duration + 1)]
        lows = [window[j][1] for j in range(duration + 1)]
        # ดึงค่า high และ low ออกมา

        close = window[duration][2]
        # ดึงค่า close ของแท่งเทียนปัจจุบันมา

        hn, ln = window[-1][0], window[-1][1]
        # ดึงค่า high และ low ในฐานะ output ที่จะใช้ฝึกพยากรณ์

        min_HL = min(highs + lows + [close])
        max_HL = max(highs + lows + [close])
        range_HL = max_HL - min_HL if max_HL - min_HL != 0 else 1e-3
        # คำนวณ min, max เพื่อจะทำ min-max scaling
        # โดยจะไม่มีการนำค่า high หรือ low จาก output มาใช้โดยเด็ดขาด

        norm_highs = [(h - min_HL) / range_HL for h in highs]
        norm_lows = [(l - min_HL) / range_HL for l in lows]
        norm_close = (close - min_HL) / range_HL
        # min-max scale ค่า high, low และ close ของ input ทั้งหมด

        x_window = []
        for nh, nl in zip(norm_highs, norm_lows):
            x_window.extend([nh, nl])
        x_window.append(norm_close)
        # นำค่าที่ถูก normalize ไปใส่ใน array สำหรับ input

        X.append([x_window])
        # นำค่าจาก array สำหรับ input ไปใส่ใน input

        nhn = (hn - min_HL) / range_HL
        nln = (ln - min_HL) / range_HL
        # min-max normalize ค่า high และ low ของ output

        y_window = [nhn, nln]
        Y.append(y_window)
        # นำค่า output ที่ถูก normalize ไปใส่ใน output

    return np.array(X), np.array(Y)
    # ส่งค่ากลับเป็น array ของ input และ output ตามลำดับ

offset_input = 0
# พยากรณ์ล่วงหน้าไปเท่าใด ในที่นี้พยากรณ์แค่แท่งเทียนถัดไปเท่านั้น

duration = 50
# ใช้ข้อมูลแท่งเทียนก่อนหน้าไปเท่าไหร่ (ไม่รวมแท่งเทียนปัจจุบัน)
# ในที่นี้คือ 50 แท่ง

X, Y = create_dataset(df.values, duration, offset_input)
# สร้าง dataset

print(X.shape, Y.shape)
# แสดงขนาดของ dataset

train_size = int(len(X) * 0.1)
# กำหนดจำนวนข้อมูลที่จะใช้ฝึก 10% (ทำให้ข้อมูลที่ใช้ทดสอบจะมี 90%)
# ด้วยการกำหนด index ที่จะใช้แบ่ง

X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]
# แบ่งข้อมูลที่จะใช้ฝึกและทดสอบตาม index ที่ได้

input_layer = Input(shape=(1, 103))
# สร้าง input layer ที่มีขนาดเป็น 1x8
# ก็คือ 1 เป็นชั้นของ layer ที่ต่ำที่สุด
# ส่วน 103 จะเป็น high 50, low 50, high 49, low 49, ..., high 0, low 0 และ close 0

x = Flatten()(input_layer)
# แปลงจากข้อมูลขนาดมิติใด ๆ เป็นเพียง 1 มิติ

output_layer = Dense(2)(x)
# เป็น dense layer ที่มีจำนวน filter เป็น 2
# สอดคล้องกับจำนวน output ที่เป็น high -1 กับ low -1

model = Model(inputs=input_layer, outputs=output_layer)
# กำหนดให้โมเดลเริ่ม input ตาม input layer และ output ตาม output layer

model.compile(optimizer='adam', loss='mean_squared_error')
# กำหนดให้โมเดลใช้ optimizer เป็น adam
# และคำนวณ loss ด้วยค่า MSE

model.summary()
# โชว์ภาพรวมของโมเดล ไม่ว่าจะเป็นขนาด layer, จำนวน parameter

model.fit(X_train, Y_train, batch_size=4, epochs=1, validation_data=(X_test, Y_test))
# ฝึกโมเดล ด้วย batch size เป็น 4, epoch เป็น 1 แล้วทดสอบความแม่นยำจากข้อมูลทดสอบ

Y_pred = model.predict(X_test, verbose=0)
# ทดสอบความสามารถในการพยากรณ์ของโมเดล
# โดย verbose เป็น 0 หมายถีง ไม่แสดงข้อความความคืบหน้าอะไรให้เห็น

r2 = r2_score(Y_test, Y_pred)
# คำนวนค่า r squared

print(f"✅ R² = {r2:.4f}")
# แสดงค่า r square ด้วยทศนิยม 4 หลัก

filename = f'IND-5mT10-ser_I50_H_L_I1_C-Flat-Dense2-O1F1H_L-b4e1-R{(r2*10000):.0f}.weights.h5'
# กำหนดชื่อไฟล์ตามลักษณะข้อมูลที่ใช้ฝึก, อัตราส่วนระหว่างที่ใช้ฝึกกับทดสอบ
# โครงสร้างของโมเดล, batch size, epoch และ r-squared ของโมเดล

model.save(filename)
# บันทึกโมเดลด้วยชื่อไฟล์นั้น ๆ

files.download(filename)
# โหลดไฟล์จาก Colab สู่คอมพิวเตอร์ของตัวเอง

def generate_pine_script_for_flatten_dense(dense_w, dense_b):
# ฟังก์ชันสร้างโค้ด Pine Script จาก weight และ bias ของโมเดล

    num_inputs, num_outputs = dense_w.shape
    # ดึงลำดับ input และลำดับ output ออกมาจากขนาดของ weight

    print("\n// === Dense Output ===")
    # แสดงข้อความให้รู้ว่าเป็น Dense Output

    for j in range(num_outputs):
    # ไล่ค่าตามจำนวน output ในที่นี้จะมี 2 ตัว คือ next high กับ next low

        terms = []
        # array เก็บส่วนย่อย ๆ ของสมการ

        for i in range(num_inputs):
        # ไล่ค่าตามจำนวน input ในที่นี้จะไล่ตั้งแต่ nh50, nl50, ... จนถึง nh0, nl0 และ nc0

            if i < 102:
                if i % 2 == 0:
                    var = f'nh{50 - i // 2}'
                else:
                    var = f'nl{50 - (i - 1) // 2}'
            else:
                var = 'nc0'
            # ตั้งชื่อตัวแปรตามลำดับของตัวแปร

            weight = dense_w[i, j]
            # ดึงค่า weight ออกมาจากลำดับ input และ output

            if abs(weight) > 1e-4:
                terms.append(f"({weight:.4f} * {var})")
            # ถ้าหาก weight มากกว่า 0.0001 ก็จะใส่ข้อความ weight * ชื่อตัวแปร

        bias = dense_b[j]
        # ดึงค่า bias ออกมาจากลำดับ output

        expr = " + ".join(terms) + f" + {bias:.4f}"
        # จัดให้อยู่ในรูปสมการ w1 * x1 + w2 * x2 + ... + bias

        print(f"out{j} = {expr}")
        # แสดงข้อความ out0 = ... ออกมา เพื่อนำไปคัดลอกใส่เป็น Pine Script

dense_w, dense_b = model.layers[2].get_weights()
# ดึง weight และ bias ของชั้น dense สุดท้ายออกมา

generate_pine_script_for_flatten_dense(dense_w, dense_b)
# รันฟังก์ชันสร้างโค้ด Pine Script
