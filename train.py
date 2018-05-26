import tensorflow as tf
from data_utils import loadDataset, get_batches, get_infer_batches
from model import TrainModel, InferenceModel
import itertools

# int_to_vocab, vocab_to_int, sample = loadDataset("data\\article_segment.txt", "data\\summarization_segment.txt")
# source_input, target_input = sample
int_to_vocab, vocab_to_int, sample = loadDataset("data\\boson_article_segment.txt", "data\\summarization_segment.txt")
source_inputs, target_inputs, target_outputs = sample
# 设置基本参数
# 词表大小
vocab_size = len(int_to_vocab)
# embedding维度
embedding_size = 128
# rnn隐藏单元数
num_units = 64
# rnn层数
num_layers = 2
# 输出最大长度
max_target_sequence_length = 60
#
max_gradient_norm = 5
# 学习率
learning_rate = 0.01
# 批次大小
batch_size = 20
# 推理每批一個句子
infer_batch_size = 2
# 训练多少代
epochs = 50
# 多少步预测一下
infer_step = 5
# beam 大小
beam_size = 5
# 分词映射
segment_to_int = vocab_to_int
# 推理模式
infer_mode = 'beam_search'


train_graph = tf.Graph()
infer_graph = tf.Graph()

with train_graph.as_default():
    train_model = TrainModel(vocab_size,embedding_size,num_units,num_layers,max_target_sequence_length,batch_size, max_gradient_norm, learning_rate)
    initializer = tf.global_variables_initializer()

with infer_graph.as_default():
    infer_model = InferenceModel(vocab_size,embedding_size,num_units,num_layers,
                                 max_target_sequence_length, infer_batch_size, beam_size, segment_to_int, infer_mode)

checkpoints_path = "model2/checkpoints"

train_sess = tf.Session(graph=train_graph)
infer_sess = tf.Session(graph=infer_graph)

train_sess.run(initializer)

infer_batch = get_infer_batches(source_inputs, infer_batch_size, vocab_to_int['<PAD>'])
print(infer_batch)
for i in itertools.count():
    for batch_i, batch in enumerate(get_batches(source_inputs, target_inputs, target_outputs, batch_size, vocab_to_int['<PAD>'], vocab_to_int['<PAD>'])):
        current_loss = train_model.train(train_sess, batch)
        print('Epoch %d Batch %d/%d - Training Loss: %f'% (i+1,batch_i+1,2000 // batch_size,current_loss) )
        if (batch_i+1) % infer_step == 0:
            print("in")
            checkpoint_path = train_model.saver.save(train_sess, checkpoints_path, global_step=(i*100 + batch_i))
            print("out")
            infer_model.saver.restore(infer_sess, checkpoint_path)
            current_predict = infer_model.infer(infer_sess, infer_batch)
            print("current_predict: ", current_predict)



