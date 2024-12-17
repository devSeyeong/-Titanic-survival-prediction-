# Titanic 데이터셋 예제: Feature Column을 활용한 모델

이 프로젝트는 **Titanic 데이터셋**과 **TensorFlow Feature Columns**를 활용하여 특성(Feature) 처리를 수행하고, 이를 바탕으로 딥러닝 모델을 구축하는 예제입니다.

---

## **1. 코드 개요**

- **데이터 처리**: Feature Columns를 활용하여 수치형, 범주형, Embedding 데이터를 정의합니다.
- **Feature Columns**: 입력 데이터를 학습 가능한 형태로 변환합니다.
- **모델 구조**: Keras Sequential API를 사용해 정규화 레이어(Normalization) 및 Dense 레이어를 추가합니다.
- **모델 학습**: 데이터 배치를 만들어 학습합니다.

---

## **2. Feature Columns 설명**

`feature_columns` 리스트에는 다양한 타입의 Feature가 정의됩니다.

### **숫자형 컬럼**

수치형 데이터를 그대로 모델에 입력합니다.

```python

feature_columns.append(tf.feature_column.numeric_column('Fare'))
feature_columns.append(tf.feature_column.numeric_column('SibSp'))
feature_columns.append(tf.feature_column.numeric_column('Parch'))

```

### **범주형 데이터 (Bucketization)**

`Age`를 기준으로 연령을 구간화(Bucketized) 합니다.

```python

Age= tf.feature_column.numeric_column('Age')
Age_bucket = tf.feature_column.bucketized_column(Age, boundaries=[10, 20, 30, 40, 50, 60, 70, 80])
feature_columns.append(Age_bucket)

```

### **범주형 데이터 (Indicator Columns)**

텍스트 또는 범주형 데이터를 **원-핫 인코딩**을 통해 모델에 입력합니다.

```python

# Sex 컬럼 처리
vocab = data['Sex'].unique()
Sex = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('Sex', vocab))
feature_columns.append(Sex)

# Embarked 컬럼 처리
vocab = data['Embarked'].unique()
Embarked = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('Embarked', vocab))
feature_columns.append(Embarked)

# Pclass 컬럼 처리
vocab = data['Pclass'].unique()
Pclass = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('Pclass', vocab))
feature_columns.append(Pclass)

```

### **Embedding Column**

카테고리의 개수가 너무 많을 때, Embedding을 사용하여 저차원 벡터로 변환합니다.

```python

vocab = data['Ticket'].unique()
Ticket = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_vocabulary_list('Ticket', vocab), dimension = 9)
feature_columns.append(Ticket)

```

---

## **3. 모델 정의**

### **모델 입력**

- 입력 데이터는 Feature Columns를 바탕으로 정규화(Normalization) 됩니다.

```python

inputs = tf.keras.layers.Input(shape=(len(feature_columns),))
norm_layer = tf.keras.layers.Normalization()(inputs)

```

### **모델 구조**

Keras Sequential API를 사용하여 3개의 주요 레이어를 추가합니다.

1. **Normalization 레이어**: 데이터 정규화
2. **Dense 레이어**: Fully Connected 레이어와 ReLU 활성화 함수
3. **Dropout 레이어**: 과적합 방지를 위한 Dropout 적용
4. **Output 레이어**: Sigmoid 활성화 함수로 이진 분류 수행

```python

model = tf.keras.Sequential([
    tf.keras.layers.Normalization(input_shape=(len(feature_columns),)),  # 정규화 레이어
    tf.keras.layers.Dense(64, activation='relu'),                       # Dense 레이어
    tf.keras.layers.Dropout(0.2),                                       # Dropout
    tf.keras.layers.Dense(1, activation='sigmoid')                      # Output 레이어
])

```

### **모델 컴파일**

- 최적화 함수: Adam
- 손실 함수: Binary Crossentropy (이진 분류)
- 평가지표: Accuracy

```python

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

```

---

## **4. 데이터 배치 및 학습**

### **데이터 배치**

데이터셋을 32개씩 묶어 배치를 만듭니다.

```python

ds_batch = ds.batch(32)

```

### **모델 학습**

- **shuffle=True**: 데이터를 무작위로 섞어 학습
- **epochs=20**: 모델이 전체 데이터를 20번 반복 학습

```python

model.fit(ds_batch, shuffle=True, epochs=20)

```

---

## **5. 실행 환경**

- **TensorFlow** 2.x 이상
- **Python** 3.8 이상
- Titanic 데이터셋 (Pandas DataFrame 형식)

---

## **6. 주의사항**

1. `data` 변수: 코드에서 `data`는 Titanic 데이터셋이 Pandas DataFrame 형태로 저장되어 있어야 합니다.
2. `ds` 변수: `ds`는 TensorFlow Dataset 형태로 미리 준비되어 있어야 합니다.
    - `tf.data.Dataset.from_tensor_slices()` 등을 사용해 Dataset으로 변환하세요.
3. `len(feature_columns)`: Feature Column의 개수에 따라 입력 차원을 조정해야 합니다.

---

## **7. 결론**

이 코드는 **TensorFlow Feature Columns**를 사용하여 Titanic 데이터셋을 기반으로 이진 분류 모델을 구축하는 예제입니다. **Feature Columns**를 통해 수치형, 범주형, Embedding 데이터를 처리하며, 정규화 및 Dropout을 추가하여 모델의 성능을 향상시키는 구조를 적용했습니다.
