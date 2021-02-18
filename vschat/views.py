from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.http import JsonResponse
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


NUM_WORDS = 1000


class QueryResult:
    label: str
    query: str
    contains_comparison: bool = False


def get_query(user_input1: str):
    max_len = 40
    vocab_size = 515
    tokenizer = Tokenizer() 

    with open('./static/word_dict_ver03.json', encoding='UTF8') as json_file:
        word_index = json.load(json_file)
        tokenizer.word_index = word_index

    # print(tokenizer.word_index)

    okt = Okt()

    tokenized_sentence = []
    temp_X = okt.morphs(user_input1, stem=True) # 토큰화
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

    result = QueryResult()
    result.label = label

    if label == '1':
        print("일주일")
        result.query = """
            SELECT * FROM stepcountData
                WHERE saved_time
                BETWEEN date('now', '-7 days', '+1 day') AND date('now')
            """
        return result

    if label == '2':
        print("주별 평균")
        result.query = """
            SELECT saved_time, cast(avg(stepCount) AS integer) AS stepCount FROM stepcountData
                WHERE saved_time BETWEEN DATE('now', 'weekday 0', '-28 days', '+1 day') AND DATE('now')
                GROUP BY strftime('%Y-%W', saved_time)
            """
        return result

    if label == '3':
        print("월별 평균")
        result.query = """
            SELECT saved_time, cast(avg(stepCount) AS integer) AS stepCount FROM stepcountData
                WHERE saved_time BETWEEN date('now', 'start of month', '-4 month', 'localtime') AND date('now', '+1 days', 'localtime')
                GROUP BY strftime('%Y-%m', saved_time)
            """
        return result

    if label == '6':
        result.contains_comparison = True
        print("주별/월별비교")
    
    else:
        print("바차트")

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

    tmp_seq = [" ".join(okt.morphs(user_input1))]
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

    result.query = "select * from stepcountData where " + predicted_seq + " ORDER BY (saved_time) ASC"
    return result


# templates 과 view 연결
def page(request):
    return render(request, 'chat.html')


@csrf_exempt
def vschat_service(request):
    if request.method != 'POST':
        return render(request, 'chat.html')

    # input1 받아옴 + 모델 탑재하고 라벨과 쿼리 받아오기
    user_input1 = request.POST['input1']

    # 유저 입력이 어떠한 종류의 query인지 판별
    result = get_query(user_input1)

    print()
    print('------------------ PRINT QUERY RESULT ------------------')
    print('* User Message=' + user_input1)
    print('* Label=' + result.label)
    print('* Query=' + result.query)
    print('* Contains Comparison=' + str(result.contains_comparison))

    try:
        data = [{
            "date": int(datetime.combine(row.saved_time, datetime.min.time()).timestamp() * 1000),
            "stepcount": row.stepCount
        } for row in StepCount_Data.objects.raw(result.query)]

        compare_with = None

        if result.contains_comparison:
            mid = (len(data) // 2) + 1
            compare_with = data[1:mid]
            data = data[mid:]

        print('* Data Length=' + str(len(data)))
        print('* Start Date=' + datetime.fromtimestamp(data[0]['date'] // 1000).strftime('%Y-%m-%d'))
        print('* End Date=' + datetime.fromtimestamp(data[-1]['date'] // 1000).strftime('%Y-%m-%d'))

        output = {
            # length field is added for debug purpose, NOT for production.
            'length': len(data),
            'data': data,
            'compare_with': compare_with,
        }

        print()
        print('------------------ PRINT OUTPUT ------------------')
        print(json.dumps(output))

        return JsonResponse(output, status=200)

    except HTTPError as e:
        print(e)

        # 500 Internal Server Error
        return JsonResponse({'message': e}, status=500)

    except IndexError as e:
        print(e)

        # 406 Not Acceptable
        return JsonResponse({'message': e}, status=406)


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