import numpy as np


# 由于文本已经分词完成，直接构建即可
def extract_character_vocab(source_data, target_data):
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    # 以行分割再以空格分割，统计分词后中文
    source_words = list(set([character for line in source_data.split('\n') for character in line.split()]))
    target_words = list(set([character for line in target_data.split('\n') for character in line.split()]))
    # 这里要把四个特殊字符添加进词典
    int_to_vocab = {idx: word for idx, word in enumerate(set(special_words + source_words + target_words))}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int


def loadDataset(source_file, target_file):
    # 读取源文件
    with open(source_file, 'r', encoding='utf-8') as f:
        source_data = f.read()
    # 读取目标文件
    with open(target_file, 'r', encoding='utf-8') as f:
        target_data = f.read()
    # 构造映射表
    int_to_vocab, vocab_to_int = extract_character_vocab(source_data, target_data)

    # 对字分词进行转换
    # 对于source来说 不需要添加<EOS>标记
    source_input = [[vocab_to_int.get(letter, vocab_to_int['<UNK>']) for letter in line.split()] for line in
                    source_data.split('\n')]
    # target_input左补<GO> target_output右补<EOS>
    target_input = [
        [vocab_to_int['<GO>']] + [vocab_to_int.get(letter, vocab_to_int['<UNK>']) for letter in line.split()] for line
        in target_data.split('\n')]

    target_output = [[vocab_to_int.get(letter, vocab_to_int['<UNK>'])
                      for letter in line.split()] + [vocab_to_int['<EOS>']] for line in target_data.split('\n')]

    return int_to_vocab, vocab_to_int, (source_input, target_input, target_output)





def get_batches(source_inputs, target_inputs, target_outputs, batch_size, source_pad_int, target_pad_int):
    def pad_sentence_batch(sentence_batch, pad_int):
        '''
        对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
        参数：
        - sentence batch
        - pad_int: <PAD>对应索引号
        '''
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    for batch_i in range(0, len(source_inputs) // batch_size):
        start_i = batch_i * batch_size
        source_inputs_batch = source_inputs[start_i: start_i + batch_size]
        target_inputs_batch = target_inputs[start_i: start_i + batch_size]
        target_outputs_batch = target_outputs[start_i: start_i + batch_size]

        pad_source_inputs_batch = np.array(pad_sentence_batch(source_inputs_batch, source_pad_int))
        pad_target_inputs_batch = np.array(pad_sentence_batch(target_inputs_batch, target_pad_int))
        pad_target_outputs_batch = np.array(pad_sentence_batch(target_outputs_batch, target_pad_int))

        targets_lengths = []
        for target in target_inputs_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in source_inputs_batch:
            source_lengths.append(len(source))

        yield pad_source_inputs_batch, pad_target_inputs_batch, pad_target_outputs_batch, source_lengths, targets_lengths


def get_infer_batches(sources, batch_size, source_pad_int):
    def pad_sentence_batch(sentence_batch, pad_int):
        '''
        对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
        参数：
        - sentence batch
        - pad_int: <PAD>对应索引号
        '''
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    sources_batch = sources[0: batch_size]
    pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
    source_lengths = []
    for source in sources_batch:
        source_lengths.append(len(source))

    return pad_sources_batch, source_lengths



