from transformers import BertModel, BertTokenizer
import torch

category_to_id_mapping_voc2012 = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19,
}


def get_embeddings(model_name='bert', dataset='VOC2012', embedding_dim=768):
    # 判断使用的模型
    if model_name == 'bert':
        model_name = 'bert-base-uncased'
        model_path = '../parameters/language_model/bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertModel.from_pretrained(model_path)
    elif model_name == 'glove':
        ...

    # 判断使用的数据集
    if dataset == 'VOC2012' or dataset == 'VOC2007':
        mapping = category_to_id_mapping_voc2012
    elif dataset == 'mscoco':
        ...

    labels = [label for label in mapping]
    max_length = embedding_dim
    embeddings = []

    # 对于每个标签，获取其BERT嵌入
    for label in labels:
        inputs = tokenizer(label, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :])

    # 将嵌入列表转换为嵌入矩阵
    embeddings_matrix = torch.stack(embeddings, dim=0)
    embeddings_matrix = embeddings_matrix.view(len(mapping), -1)
    return embeddings_matrix


if __name__ == '__main__':
    embeddings = get_embeddings()
    print(embeddings.shape)
