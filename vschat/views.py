from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from .models import StepCount_Data
from urllib.error import HTTPError
# 해당 모델 관련 분류하기 및 주석 처리
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import konlpy
from konlpy.tag import *

# Create your views here.

NUM_WORDS = 1000

# 날짜와 걸음 수 저장 변수
xValue = []
yValue = []
legend_value = []
week = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일','일요일']
x1 = []
x2 = []
y1 = []
y2 = []
response = []

# templates 과 view 연결
def page(request):
    return render(request, 'chat.html')

@csrf_exempt
def vschat_service(request):
    # text 입력 받은 후
    if request.method == 'POST':
        
        # input1 받아옴 + 모델 탑재하고 라벨과 쿼리 받아오기
        input1 = request.POST['input1']

        okt = Okt()

        max_len = 40

        vocab_size = 515
        tokenizer = Tokenizer() 


        with open('./static/word_dict_ver03.json', encoding='UTF8') as json_file:
            word_index = json.load(json_file)
            tokenizer.word_index = word_index

        # print(tokenizer.word_index)

        tokenized_sentence = []
        temp_X = okt.morphs(input1, stem=True) # 토큰화
        tokenized_sentence.append(temp_X)
        print(tokenized_sentence)

        input_data = tokenizer.texts_to_sequences(tokenized_sentence)
        print(input_data)

        input_data = pad_sequences(input_data, maxlen=max_len) # padding

        loaded_model = load_model('./static/best_model_ver_relu_epc500.h5')
        prediction = loaded_model.predict(input_data)
        print(prediction)
        print("label: ", np.argmax(prediction[0]))

        label = str(np.argmax(prediction[0]))

        if label == '1':
            query = "select * from stepcountData where saved_time BETWEEN date('now', '-7 days', 'localtime') AND date('now', 'localtime');"
        elif label == '2':
            query = "select * from stepcountData where saved_time BETWEEN date('now', '-35 days',  'localtime') AND date('now', 'localtime');"
        elif label == '3':
            query = "select * from stepcountData where saved_time BETWEEN date('now', '-4 months','start of month', 'localtime') AND date('now', '+1 days', 'localtime');"
        else:
                
            with open('./static/tokenizer_for_attention.json', encoding='UTF8') as f:
                data = json.load(f)
                tokenizer = tokenizer_from_json(data)
                
            # 모델 생성
            model = Seq2seq(sos=tokenizer.word_index['\t'], eos=tokenizer.word_index['\n'])

            model.load_weights("./static/attention_ckpt/attention_ckpt")

            # Implement algorithm test
            @tf.function
            def test_step(model, inputs):
                return model(inputs, training=False)


            tmp_seq = [" ".join(okt.morphs(input1))]
            print("tmp_seq : ", tmp_seq)

            test_data = list()
            test_data = tokenizer.texts_to_sequences(tmp_seq)
            print("tokenized data : ", test_data)

            prd_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,value=0,padding='pre',maxlen=128)

            prd_data = tf.data.Dataset.from_tensor_slices(prd_data).batch(1).prefetch(1024)

            for seq in prd_data :
                prediction = test_step(model, seq)
                
                predicted_seq = tokenizer.sequences_to_texts(prediction.numpy())
                print(predicted_seq)
                print("predict tokens : ", prediction.numpy())

            predicted_seq = str(predicted_seq[0]).replace(" _ ", "_")
            predicted_seq = predicted_seq.replace("e (", "e(")
            predicted_seq = predicted_seq.replace("' ", "'")
            predicted_seq = predicted_seq.replace(" '", "'")
            predicted_seq = predicted_seq.replace(" - ", "-")
            predicted_seq = predicted_seq.replace("+ ", "+")
            predicted_seq = predicted_seq.replace("- ", "-")
            print(predicted_seq)
            query = "select * from stepcountData where " + predicted_seq + ";"

        
        # if not legend_value or xValue or yValue or response:
        legend_value.clear()
        xValue.clear()
        yValue.clear()
        response.clear()
        x1.clear()
        x2.clear()
        y1.clear()
        y2.clear()

        print(legend_value, xValue, yValue, response, x1, x2, y1, y2)

        try:
            if label == "2":
                show_weeks_avg(query)
                print("주별 평균")
            elif label == "3":
                show_months_avg(query)
                print("월별 평균")
            elif label == "6":
                if check_week(query) == True:
                    show_by_week(query)
                    print('주별비교')
                else:
                    show_by_month(query)
                    print('월별비교')

            else:
                show_barchart(query)
                print('바차트')
        except HTTPError as e:
            print("httperror")
            print("데이터를 불러올 수 없습니다. 텍스트를 다시 입력하세요")
        except IndexError as e:
            print("indexerror")
            print("데이터를 불러올 수 없습니다. 텍스트를 다시 입력하세요")
        # 예외처리에 대한 알림 메세지 어떻게 출력하는지 보기

        # 딕셔너리에 저장(응답, 쿼리 결과 저장 변수, 라벨)
        output = dict()
        if not output:
            # output['response'] = response
            output['response'] = "그래프가 출력되었습니다"
            output['xValues'] = xValue
            output['yValues'] = yValue
            output['label'] = label
            output['legend_value'] = legend_value
            print(output)

        else: 
            del output
            output['response'] = response
            output['response'] = "그래프가 출력되었습니다"
            output['xValues'] = xValue
            output['yValues'] = yValue
            output['label'] = label
            output['legend_value'] = legend_value
            print(output)

        print("-----------------------------------------")
        print("-----------------------------------------")

        return HttpResponse(json.dumps(output), status=200)

    else:
        return render(request, 'chat.html')


## 함수 및 클래스 ------------------------------------------------------
def check_week(query):
    print(str(query))
    if 'weekday' in str(query):
        return True
    else:
        return False

def show_weeks_avg(query):
    check_num = 0
    for p in StepCount_Data.objects.raw(query): 
        date = pd.to_datetime(p.saved_time)
        print("date", date)
    
        if check_num in [0, 7, 14, 21, 28]:
            x1.append(str(date.month) + '월 ' + str(date.day) + '일 ~ ')
        elif check_num in [6, 13, 20, 27, 34]:
            x2.append(str(date.month) + '월 ' + str(date.day) + '일')

        y1.append(p.stepCount)

        check_num = check_num + 1

    for i in range(5):
        xValue.append(x1[i] + x2[i])
        yValue.append(sum(y1[i * 7 : i * 7 + 7])/7)
        

    response.append("최근 5주 주별 평균입니다.")
    legend_value.append(" ")

    return xValue, yValue, response, legend_value

def show_months_avg(query):
    for p in StepCount_Data.objects.raw(query):
        date = pd.to_datetime(p.saved_time)
        print("date", date)

        if not xValue:
            for i in range(5):
                month = date.month + i
                xValue.append(str(month) + "월")

        y1.append(p.stepCount)
        
    
    for i in range(5):
        cnt = i * 30
        if cnt == 120:
            div = len(y1) - 119
            yValue.append(sum(y1[cnt:])/div)
        elif cnt < 120:
            yValue.append(sum(y1[cnt:cnt+30])/30)      
    
    print("길이", len(yValue))
    
    response.append("최근 다섯 달 평균입니다.")
    legend_value.append(" ")

    return xValue, yValue, response, legend_value

def show_barchart(query):
    for p in StepCount_Data.objects.raw(query):
        # print(p.saved_time)
        # 판다스로 날짜 가져옴
        date = pd.to_datetime(p.saved_time)
        print("date", date)

        # 범례 비었으면 범례에 월 넣어줌
        if not legend_value:
            legend_value.append(str(date.month))
            print(legend_value)

        # 일과, 걸음수 리스트에 저장
        xValue.append(date.day)
        # print(p.stepCount)
        yValue.append(p.stepCount)

    response.append(legend_value[0] + "월 그래프입니다.")

    return xValue, yValue, response, legend_value

def show_by_week(query):
    for p in StepCount_Data.objects.raw(query):
        
        # 판다스 날짜로 변환, 요일을 추출하여 저장
        date = pd.to_datetime(p.saved_time)
        day = date.weekday()

        # 일주일 데이터가 들어왔을 때 범례 저장, 데이터 저장
        if len(x1) == 7:
            if legend_value[1] == -1:
                start = str(date).replace(" 00:00:00","")
                end = str(date + pd.Timedelta(days = 6)).replace(" 00:00:00","")
                legend_value[1] = start + ' ~ ' + end

            x2.append(week[day])
            y2.append(p.stepCount)
        
        # 첫 일주일 데이터 범례, 데이터 저장
        else:
            if not legend_value:
                start = str(date).replace(" 00:00:00","")
                end = str(date + pd.Timedelta(days = 6)).replace(" 00:00:00","")
                legend_value.append(start + ' ~ ' + end)
                legend_value.append(-1)

            x1.append(week[day])
            y1.append(p.stepCount)

    # 데이터 한 변수에 저장
    xValue.append(x1)
    xValue.append(x2)
    # print(p.stepCount)
    yValue.append(y1)
    yValue.append(y2)

    response.append(legend_value[0] + "과 " + legend_value[1] + " 비교입니다.")

    return xValue, yValue, response, legend_value
        

def show_by_month(query):
    for p in StepCount_Data.objects.raw(query):
        # print(p.saved_time)
        # 판다스 날짜로 변환
        date = pd.to_datetime(p.saved_time)

        # 월을 변수에 저장(첫번째 월)
        if not legend_value:  
            legend_value.append(str(date.month))
            legend_value.append('none')
            print(legend_value)
            # 걸음수와 날짜 저장
            x1.append(date.day)
            y1.append(p.stepCount)

        # 범례에 값이 있을 때
        else: 
            # 범례에 있는 값과 월이 같을 때
            if legend_value[0] == str(date.month):
                x1.append(date.day)
                y1.append(p.stepCount)

            # 범례에 있는 값과 월이 다를 때
            else:
                # 월 범례에 저장
                if legend_value[1] == 'none':
                    legend_value[1] = str(date.month)

                # 새 변수에 날짜와 걸음수 저장
                x2.append(date.day)
                y2.append(p.stepCount)
                
    # 범례 01월 같은 달은 0을 빼기
    for i in range(len(legend_value)):
        legend_value[i] = legend_value[i] + " 월"

    xValue.append(x1)
    xValue.append(x2)
    # print(p.stepCount)
    yValue.append(y1)
    yValue.append(y2)

    response.append(legend_value[0] + "과 " + legend_value[1] + " 비교입니다.")

    return xValue, yValue, response, legend_value

class Encoder(tf.keras.Model):
  def __init__(self):
    super(Encoder, self).__init__()
    # 1000개의 단어들을 128크기의 vector로 Embedding해줌.
    self.emb = tf.keras.layers.Embedding(NUM_WORDS, 128)
    # return_state는 return하는 Output에 최근의 state를 더해주느냐에 대한 옵션
    # 즉, Hidden state와 Cell state를 출력해주기 위한 옵션이라고 볼 수 있다.
    # default는 False이므로 주의하자!
    # return_sequence=True로하는 이유는 Attention mechanism을 사용할 때 우리가 key와 value는
    # Encoder에서 나오는 Hidden state 부분을 사용했어야 했다. 그러므로 모든 Hidden State를 사용하기 위해 바꿔준다.
    self.lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)

  def call(self, x, training=False, mask=None):
    x = self.emb(x)
    H, h, c = self.lstm(x)
    return H, h, c


class Decoder(tf.keras.Model):
  def __init__(self):
    super(Decoder, self).__init__()
    self.emb = tf.keras.layers.Embedding(NUM_WORDS, 128)
    # return_sequence는 return 할 Output을 full sequence 또는 Sequence의 마지막에서 출력할지를 결정하는 옵션
    # False는 마지막에만 출력, True는 모든 곳에서의 출력
    self.lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)
    # LSTM 출력에다가 Attention value를 dense에 넘겨주는 것이 Attention mechanism이므로
    self.att = tf.keras.layers.Attention()
    self.dense = tf.keras.layers.Dense(NUM_WORDS, activation='softmax')

  def call(self, inputs, training=False, mask=None):
    # x : shifted output, s0 : Decoder단의 처음들어오는 Hidden state
    # c0 : Decoder단의 처음들어오는 cell state H: Encoder단의 Hidden state(Key와 value로 사용)
    x, s0, c0, H = inputs
    x = self.emb(x)

    # initial_state는 셀의 첫 번째 호출로 전달 될 초기 상태 텐서 목록을 의미
    # 이전의 Encoder에서 만들어진 Hidden state와 Cell state를 입력으로 받아야 하므로
    # S : Hidden state를 전부다 모아놓은 것이 될 것이다.(Query로 사용)
    S, h, c = self.lstm(x, initial_state=[s0, c0])

    # Query로 사용할 때는 하나 앞선 시점을 사용해줘야 하므로
    # s0가 제일 앞에 입력으로 들어가는데 현재 Encoder 부분에서의 출력이 batch 크기에 따라서 length가 현재 1이기 때문에 2차원형태로 들어오게 된다.
    # 그러므로 이제 3차원 형태로 확장해 주기 위해서 newaxis를 넣어준다.
    # 또한 decoder의 S(Hidden state) 중에 마지막은 예측할 다음이 없으므로 배제해준다.
    S_ = tf.concat([s0[:, tf.newaxis, :], S[:, :-1, :]], axis=1)

    # Attention 적용
    # 아래 []안에는 원래 Query, Key와 value 순으로 입력해야하는데 아래처럼 두가지만 입력한다면
    # 마지막 것을 Key와 value로 사용한다.
    A = self.att([S_, H])

    y = tf.concat([S, A], axis=-1)
    return self.dense(y), h, c


class Seq2seq(tf.keras.Model):
  def __init__(self, sos, eos):
    super(Seq2seq, self).__init__()
    self.enc = Encoder()
    self.dec = Decoder()
    self.sos = sos
    self.eos = eos

  def call(self, inputs, training=False, mask=None):
    if training is True:
      # 학습을 하기 위해서는 우리가 입력과 출력 두가지를 다 알고 있어야 한다.
      # 출력이 필요한 이유는 Decoder단의 입력으로 shited_ouput을 넣어주게 되어있기 때문이다.
      x, y = inputs

      # LSTM으로 구현되었기 때문에 Hidden State와 Cell State를 출력으로 내준다.
      H, h, c = self.enc(x)

      # Hidden state와 cell state, shifted output을 초기값으로 입력 받고
      # 출력으로 나오는 y는 Decoder의 결과이기 때문에 전체 문장이 될 것이다.
      y, _, _ = self.dec((y, h, c, H))
      return y

    else:
      x = inputs
      H, h, c = self.enc(x)

      # Decoder 단에 제일 먼저 sos를 넣어주게끔 tensor화시키고
      y = tf.convert_to_tensor(self.sos)
      # shape을 맞춰주기 위한 작업이다.
      y = tf.reshape(y, (1, 1))

      # 최대 64길이 까지 출력으로 받을 것이다.
      seq = tf.TensorArray(tf.int32, 128)

      # tf.keras.Model에 의해서 call 함수는 auto graph모델로 변환이 되게 되는데,
      # 이때, tf.range를 사용해 for문이나 while문을 작성시 내부적으로 tf 함수로 되어있다면
      # 그 for문과 while문이 굉장히 효율적으로 된다.
      for idx in tf.range(128):
        y, h, c = self.dec([y, h, c, H])
        # 아래 두가지 작업은 test data를 예측하므로 처음 예측한값을 다시 다음 step의 입력으로 넣어주어야하기에 해야하는 작업이다.
        # 위의 출력으로 나온 y는 softmax를 지나서 나온 값이므로
        # 가장 높은 값의 index값을 tf.int32로 형변환해주고
        # 위에서 만들어 놓았던 TensorArray에 idx에 y를 추가해준다.
        y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.int32)
        # 위의 값을 그대로 넣어주게 되면 Dimension이 하나밖에 없어서
        # 실제로 네트워크를 사용할 때 Batch를 고려해서 사용해야 하기 때문에 (1,1)으로 설정해 준다.
        y = tf.reshape(y, (1, 1))
        seq = seq.write(idx, y)

        if y == self.eos:
          break
      # stack은 그동안 TensorArray로 받은 값을 쌓아주는 작업을 한다.    
      return tf.reshape(seq.stack(), (1, 128))