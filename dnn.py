import pandas as pd
import tensorflow as tf


data_train = pd.read_csv('train_1.csv')
print(data_train.head())
data_train = data_train.ix[:, 2:]

data_test = pd.read_csv('test_1.csv')
# print(test.head())
data_test = data_test.ix[:, 2:]


train_x, train_y = data_train, data_train.pop('current_service')
test_x, test_y = data_test, data_test.pop('current_service')

my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
# print(my_feature_columns)

classifier = tf.estimator.DNNClassifier(
    # 这个模型接受哪些输入的特征    
    feature_columns=my_feature_columns,
    # 包含两个隐藏层，每个隐藏层包含10个神经元.
    hidden_units=[32, 64,32],
    # 最终结果要分成的几类
    n_classes=11)

def train_func(train_x,train_y):
    dataset=tf.data.Dataset.from_tensor_slices((dict(train_x), train_y))
    dataset = dataset.shuffle(1000).repeat().batch(100)
    return dataset

classifier.train(
    input_fn=lambda: train_func(train_x, train_y),
    steps=1000)

def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset

predict_arr = []
predictions = classifier.predict(input_fn=lambda: eval_input_fn(test_x, labels=test_y, batch_size=100))
for predict in predictions:
    predict_arr.append(predict['probabilities'].argmax())

result = predict_arr == test_y
result1 = [w for w in result if w == True]
print("准确率为 %s" % str((len(result1) / len(result))))

