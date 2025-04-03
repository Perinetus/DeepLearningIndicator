'''
MIT License

Copyright (c) 2025 Natchaphol Chumpanin (Account name as Perinetus)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

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

file_id = "1IXNJZGV9Jvg2pvMYIGO98ajUSxS8l3nf"
# กำหนดค่าคงตัว file_id เป็น id ของไฟล์ Google Drive ซึ่งจะมาจาก
# https://drive.google.com/file/d/1IXNJZGV9Jvg2pvMYIGO98ajUSxS8l3nf/view?usp=drive_link

# ข้อมูลจะเป็นกราฟแท่งเทียน 1 ชั่วโมง ของ XAU/USD
# ข้อมูลนี้ได้มาจากใน Kaggle ชื่อ "XAU/USD Gold Price Historical Data (2004-2024)" ของ Novandra Anugrah

# ช่วงเวลาของข้อมูลจะอยู่ ระหว่าง 7 โมงเช้าของวันที่ 11 มิ.ย. 2547
# จนถึงตี 2 ของวันที่ 20 กันยายน 2567

csv_path = "gold_prices.csv"
# กำหนดค่าคงตัวเป็นชื่อไฟล์ที่จะบันทึกลงไปใน Colab

gdown.download(f"https://drive.google.com/uc?id={file_id}", csv_path, quiet=False)
# ดาวน์โหลดไฟล์จาก Google Drive แล้วสร้างไฟล์บน Colab

df = pd.read_csv(csv_path)
# อ่านไฟล์ csv ไปใส่ในตัวแปร df

# ตัวของไฟล์ csv จะมีลักษณะข้อมูลประมาณนี้
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

df = df[['High', 'Low', 'Volume']]
# ดึงข้อมูลเฉพาะ high, low และ volume ออกมา

def create_dataset(data, offset_input):
# ฟังก์ชัน label ข้อมูลจากข้อมูลที่ดึงออกมา
    X = []
    Y = []
    # สร้าง array เพื่อใช้เป็น input (X) และ output (Y)

    showing = 3
    # มีไว้ debug ตัวโค้ดโดยจะแสดงข้อมูลข้างใน 3 ครั้ง

    for i in range(len(data) - offset_input - 3):
    # ไล่อ่านข้อมูลตามจำนวนข้อมูล ลบด้วย offset กับ 3 ที่เป็นขนาดของข้อมูลที่ใช้ฝึกกับทดสอบ
    # เพื่อไม่ให้เกิด error จากการไล่ index เกินขนาดของข้อมูล

        window = data[i:i + offset_input + 3]
        # สร้างค่าคงตัว window ที่มีช่วงข้อมูลที่จะใช้เป็น input กับ output

        # โครงสร้างของ window จะมีลักษณะดังนี้
        # [[high 1    low 1    volume 1 ]
        #  [high 0    low 0    volume 0 ]
        #  [high -1   low -1   volume -1]]
        # โดยที่ 1 จะเป็นข้อมูลจากแท่งเทียนก่อนหน้า
        # 0 จะเป็นข้อมูลจากแท่งเทียนปัจจุบัน
        # และ -1 จะเป็นข้อมูลจากแท่งเทียนถัดไป

        h1 = window[0][0]
        l1 = window[0][1]
        v1 = window[0][2]
        h0 = window[1][0]
        l0 = window[1][1]
        v0 = window[1][2]
        hn = window[2][0]
        ln = window[2][1]
        # เก็บค่าไปใส่ในค่าคงตัวชื่อต่าง ๆ เพื่อให้เรียกใช้งานได้ง่าย

        min_HL = min(h1, l1, h0, l0)
        max_HL = max(h1, l1, h0, l0)
        range_HL = max_HL - min_HL
        nh1 = (h1 - min_HL) / max(range_HL, 1e-3)
        nl1 = (l1 - min_HL) / max(range_HL, 1e-3)
        nh0 = (h0 - min_HL) / max(range_HL, 1e-3)
        nl0 = (l0 - min_HL) / max(range_HL, 1e-3)
        nhn = (hn - min_HL) / max(range_HL, 1e-3)
        nln = (ln - min_HL) / max(range_HL, 1e-3)
        # normalize ค่าด้วย min max scaling
        # โดยค่าที่อ้างอิง จะไม่รวม output เพื่อไม่ให้ข้อมูลรั่วไหล

        x_window = []
        # สร้าง array ไว้สำหรับ input

        x_window.append(nh1)
        x_window.append(nl1)
        x_window.append(nh0)
        x_window.append(nl0)
        # ใส่ high 1, low 1, high 0 และ low 0 ลงใน input ตามลำดับ

        volume_change = 1 if v0 < v1 else (0 if v0 > v1 else 0.5)
        # ถ้าหาก volume มีค่าสูงขึ้น
        # จะกำหนดค่าคงตัว volume change เป็น 1 ในฐานะค่าความจริงเป็นจริง
        # มิฉะนั้น จะกำหนดค่าเป็น 0 ในฐานะค่าความจริงเป็นเท็จ
        # แต่ถ้าหาก volume ไม่เปลี่ยนแปลง จะกำหนดค่าเป็น 0.5 ในฐานะที่กึ่งจริงกึ่งเท็จ

        x_window.append(volume_change)
        # นำค่า window change ไปใส่ใน array สำหรับ input

        if volume_change == 1:
            ratio = v1 / max(1, v0)
        elif volume_change == 0:
            ratio = v0 / max(1, v1)
        else:
            ratio = 1
        # หาอัตราส่วน โดยจะคิดจาก volume น้อย / volume มากเสมอ
        # ถ้าหากเกิดตัวหารเป็น 0 (มีโอกาสน้อย) ก็จะกำหนดให้ตัวหารไม่ใช่ 0

        x_window.append(ratio)
        # นำค่าอัตราส่วน ไปใส่ใน array สำหรับ input

        X.append([x_window])
        # นำค่าจาก array สำหรับ input ไปใส่ใน input
        # โดยจะต้องครอบด้วย [] เพื่อให้ขนาดของ input เป็น (1, 6) แทนที่จะเป็น (6)
        # มิฉะนั้นจะรันผ่าน convolution layer ไม่ได้

        nhn = (min(max(nhn, -0.5), 1.5) + 0.5) / 2
        nln = (min(max(nln, -0.5), 1.5) + 0.5) / 2
        # Output จะกำหนดให้อยู่ในช่วงระหว่าง -0.5 ถึง 1.5
        # ซึ่งจะ normalize ให้อยู่ในช่วง 0 กับ 1

        y_window = []
        # สร้าง array ไว้สำหรับ output

        y_window.append(nhn)
        y_window.append(nln)
        # ใส่ high -1 และ low -1 ลงไปใน array สำหรับ output

        Y.append(y_window)
        # นำค่าจาก array สำหรับ output ไปใส่ใน output

        if showing > 0:
            print(window, "\n")
            print(x_window, "\n")
            print(y_window, "\n")
            showing -= 1
        # แสดงค่าต่าง ๆ เพื่อ debug ตัวโค้ด

    return np.array(X), np.array(Y)
    # return ค่าเป็นชุดของ input กับ output

offset_input = 0
# การเลื่อน ความหมายของมันก็คือ พยากรณ์ล่วงหน้าถัดไปเท่าไหร่
# อย่าง offset เป็น 0 จะหมายถึงพยากรณ์แค่แท่งเทียนถัดไป
# แต่ถ้า offset เป็น 1 ก็จะหมายถึงพยากรณ์แค่แท่งเทียนถัดไปอีก 1 แท่ง เป็นต้น

X, Y = create_dataset(df.values, offset_input)
# สร้างชุดข้อมูลฝึกและทดสอบ

print(X.shape, Y.shape)
# แสดงขนาดของชุดของ input กับ output เพื่อ debug

train_size = int(len(X) * 0.5)
# กำหนดจำนวนข้อมูลที่จะใช้ฝึก 50% (ทำให้ข้อมูลที่ใช้ทดสอบจะมี 50%)
# ด้วยการกำหนด index ที่จะใช้แบ่ง

X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]
# แบ่งข้อมูลที่จะใช้ฝึกและทดสอบตาม index ที่ได้

input_layer = Input(shape=(1, 6))
# สร้าง input layer ที่มีขนาดเป็น 1x6
# ก็คือ 1 เป็นชั้นของ layer ที่ต่ำที่สุด
# ส่วน 6 จะเป็น high 1, low 1, high 0, low 0, volume change และ ratio

x = Conv1D(filters=4, kernel_size=1)(input_layer)
# ต่อกับ convolution layer ที่มีจำนวน filter เป็น 4
# มี kernel size เป็น 1 (เพราะมีชั้นของ layer ก่อนหน้าเพียง 1)

x = Activation('relu')(x)
# ตามด้วย activation function ที่เป็น ReLU
# กล่าวคือ ถ้าค่า input ไม่น้อยกว่า 0 ค่า output ก็จะเป็นตาม input
# แต่ถ้าค่า input น้อยกว่า 0 output จะกลายเป็น 0

x = Flatten()(x)
# แปลงจากข้อมูลขนาดมิติใด ๆ เป็นเพียง 1 มิติ

x = Dense(2)(x)
# เป็น dense layer ที่มีจำนวน filter เป็น 2
# สอดคล้องกับจำนวน output ที่เป็น high -1 กับ low -1

output_layer = Activation('relu')(x)
# สุดท้าย output layer จะเป็น activation function ที่เป็น ReLU ต่อจาก layer ก่อนหน้า

model = Model(inputs=input_layer, outputs=output_layer)
# กำหนดให้โมเดลเริ่ม input ตาม input layer และ output ตาม output layer

model.compile(optimizer='adam', loss='mean_squared_error')
# กำหนดให้โมเดลใช้ optimizer เป็น adam
# และคำนวณ loss ด้วยค่า MSE

model.summary()
# โชว์ภาพรวมของโมเดล ไม่ว่าจะเป็นขนาด layer, จำนวน parameter

model.fit(X_train, Y_train, batch_size=4, epochs=10, validation_data=(X_test, Y_test))
# ฝึกโมเดล ด้วย batch size เป็น 4, epoch เป็น 10 แล้วทดสอบความแม่นยำจากข้อมูลทดสอบ

Y_pred = model.predict(X_test, verbose=0)
# ทดสอบความสามารถในการพยากรณ์ของโมเดล
# โดย verbose เป็น 0 หมายถีง ไม่แสดงข้อความความคืบหน้าอะไรให้เห็น

r2 = r2_score(Y_test, Y_pred)
# คำนวนค่า r squared

print(f"✅ R² = {r2:.4f}")
# แสดงค่า r square ด้วยทศนิยม 4 หลัก

filename = f'IND-1hT50-ser_I2_H_L_chmV_ratioV_actNorm-Conv4_1-ReLU-Flat-Dense2-ReLU-O1F0H_L-b4e10-R{(r2*10000):.0f}.weights.h5'
# กำหนดชื่อไฟล์ตามลักษณะข้อมูลที่ใช้ฝึก, อัตราส่วนระหว่างที่ใช้ฝึกกับทดสอบ
# โครงสร้างของโมเดล, batch size, epoch และ r-squared ของโมเดล

model.save(filename)
# บันทึกโมเดลด้วยชื่อไฟล์นั้น ๆ

files.download(filename)
# โหลดไฟล์จาก Colab สู่คอมพิวเตอร์ของตัวเอง

def generate_pine_script_for_conv_dense(model_name, conv_w, conv_b, dense_w, dense_b):
    # ฟังก์ชันนี้ จะมีหน้าที่สร้างโค้ด Pine script ตาม weight และ bias ของโมเดล
    # โดยจะมีชื่อโมเดล เผื่อสร้างโมเดลหลายตัว ป้องกันการตั้งชื่อตัวแปรซ้ำกัน

    num_steps, num_feats, num_filters = conv_w.shape
    # ดึงจำนวน step, จำนวน feature และ จำนวน filter ออกมาจากขนาดของ convolution layer

    print("// === Conv1D Output with ReLU ===")
    # แสดงข้อความให้รู้ว่า จะเป็นโค้ด convolution layer
    # ที่มี activation function เป็น ReLU

    for i in range(num_filters):
    # ไล่ตัวแปร i ตามจำนวน filter

        terms = []
        # สร้าง array ชื่อ terms ไว้เก็บ weight * input แต่ละตัว

        for step in range(num_steps):
        # ไล่ตัวแปร step ตามจำนวน step

            for feat in range(num_feats):
            # ไล่ตัวแปร feat ตามจำนวน feature
                if feat == 0:
                    var = 'nh1'
                elif feat == 1:
                    var = 'nl1'
                elif feat == 2:
                    var = 'nh0'
                elif feat == 3:
                    var = 'nl0'
                elif feat == 4:
                    var = 'ivi'
                elif feat == 5:
                    var = 'nvc'
                # กำหนดชื่อตัวแปรต่าง ๆ ให้สอดคล้องกับโค้ด Pine Script ตามด้วยลำดับของ feature ในฐานะ input

                weight = conv_w[step][feat][i]
                # กำหนด weight ตามค่า weight ใน convolution layer นั้น ๆ

                terms.append(f"({weight:.8f} * {var})")
                # ต่อข้อความให้กลายเป็น weight * input

        bias = conv_b[i]
        # ใส่ค่า bias ตาม convolution layer นั้น ๆ

        expr = " + ".join(terms) + f" + {bias:.8f}"
        # จัดให้โค้ด convolution layer นั้น ๆ อยู่ในรูป
        # weight0 * input0 + weight1 * input1 + ...

        print(f"{model_name}_f{i} = math.max({expr}, 0)")
        # ครอบด้วย math.max(convolution layer, 0)
        # ในฐานะ activation function ที่เป็น ReLU

    print("\n// === Dense Output with ReLU ===")
    # แสดงข้อความให้รู้ว่า จะเป็นโค้ด dense layer
    # ซึ่งก็มี activation function เป็น ReLU เหมือนกัน

    for j in range(dense_w.shape[1]):
    # ไล่ตัวแปร j ตามขนาดมิติที่ 2 ของ dense layer

        terms = []
        # สร้าง array ชื่อ terms ไว้เก็บ weight * ค่าที่ได้จาก layer ก่อนหน้า

        for i in range(dense_w.shape[0]):
        # ไล่ตัวแปร i ตามขนาดมิติที่ 1 ของ dense layer

            weight = dense_w[i][j]
            # กำหนด weight ตามค่า weight ใน dense layer นั้น ๆ

            terms.append(f"({weight:.8f} * {model_name}_f{i})")
            # ต่อข้อความให้กลายเป็น weight * ค่าที่ได้จาก layer ก่อนหน้า

        bias = dense_b[j]
        # ใส่ค่า bias ตาม dense layer นั้น ๆ

        expr = " + ".join(terms) + f" + {bias:.8f}"
        # จัดให้โค้ด dense layer นั้น ๆ อยู่ในรูป
        # weight0 * conv0 + weight1 * conv1 + ...

        print(f"{model_name}_out{j} = math.max({expr}, 0)")
        # ครอบด้วย math.max(convolution layer, 0)
        # ในฐานะ activation function ที่เป็น ReLU เช่นกัน

conv_w, conv_b = model.layers[1].get_weights()
# ดึง weight กับ bias ของ convolution layer ออกมา

dense_w, dense_b = model.layers[4].get_weights()
# ดึง weight กับ bias ของ dense layer ออกมา

generate_pine_script_for_conv_dense("HLchmV0FactNorm", conv_w, conv_b, dense_w, dense_b)
# สร้าง Pine Script ไว้สำหรับรันบน TradingView
