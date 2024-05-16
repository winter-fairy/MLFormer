import torch.nn as nn
import torch
import timm
import math
from model.MyViT import MyViT
from model.embedding import get_embeddings


class MLFormer(nn.Module):
    def __init__(self, depth=12, num_class=20, embed_dim=768, num_heads=8, hidden_dim=768 * 2, start_layer=8,
                 attention_dropout_prob=0.1, proj_dropout_prob=0.1, ff_dropout_prob=0.1,
                 model_name='vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=False, embed_type='random'):
        """

        :param depth: 网络的层数，即深度;最大值为12，因为目前采用的ViT模型最多12层，
        :param num_class: 最终分类个数
        :param embed_dim: 嵌入维度
        :param num_heads: 头数
        :param hidden_dim: 注意力机制中FNN隐藏层维度
        :param start_layer: 从哪一层开始text stream
        :param attention_dropout_prob:
        :param proj_dropout_prob:
        :param ff_dropout_prob:
        :param model_name: ViT模型
        :param pretrained: 是否使用预训练模型
        """
        super().__init__()
        self.depth = depth
        self.num_class = num_class
        self.start_layer = start_layer

        # 标签嵌入
        self.embed_type = embed_type
        if embed_type != 'random':
            self.text_feature = get_embeddings(model_name=embed_type).cuda()
            self.embedding = nn.Parameter(self.text_feature)
        else:
            self.text_feature = torch.eye(self.num_class).cuda() # 为每一个标签生成标签嵌入  [20,20]
        self.text_linear = nn.Linear(self.num_class, embed_dim)

        # img branch
        self.img_branch = MyViT(model_name=model_name, pretrained=pretrained)

        # text_branch
        self.text_branch = [TextBlock(embed_dim, num_heads, hidden_dim, attention_dropout_prob,
                                              proj_dropout_prob, ff_dropout_prob).cuda() for _ in range(depth)]

        # 分类头
        self.head = TextHead(embed_dim, num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_features):
        batch_size = img_features[0].size(0)

        # 得到标签嵌入
        if self.embed_type == 'random':
            text_features = torch.stack([self.text_feature for _ in range(batch_size)], dim=0)  # [bs,20,20]
            text_features = self.text_linear(text_features)  # [bs,20,embed_dim]
        else:
            text_features = torch.stack([self.text_feature for _ in range(batch_size)], dim=0)  # [bs,20,768]

        # text_branch前向传播
        for i in range(self.start_layer, self.depth):
            tmp_img_features = img_features[i]  # 当前层图像分支的输出
            text_features = self.text_branch[i](tmp_img_features, text_features)

        # 经过分类头
        output = self.head(text_features)
        output = output.view(batch_size, -1)
        return output


class TextBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, attention_dropout_prob=0.1, proj_dropout_prob=0.1,
                 ff_dropout_prob=0.1):
        super(TextBlock, self).__init__()
        self.text_norm = nn.LayerNorm(embed_dim)
        self.img_norm = nn.LayerNorm(embed_dim)
        self.attention1 = MultiHeadAttention(embed_dim, num_heads, attention_dropout_prob, proj_dropout_prob)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.attention2 = MultiHeadAttention(embed_dim, num_heads, attention_dropout_prob, proj_dropout_prob)

        self.norm3 = nn.LayerNorm(embed_dim)
        self.FFN = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Dropout(ff_dropout_prob),
                                 nn.Linear(hidden_dim, embed_dim))

    def forward(self, img_features, text_features):
        # 第一次注意力机制
        # print(text_features.device, img_features.device)
        text_features = self.text_norm(text_features)  # norm
        img_features = self.img_norm(img_features)
        x = self.attention1(img_features, text_features)
        text_features = text_features + x  # add

        # 第二次注意力机制，qkv都是来自text_features
        text_features = self.norm2(text_features)
        x = self.attention2(text_features, text_features)
        text_features = text_features + x

        # FFN
        text_features = self.norm3(text_features)
        x = self.FFN(text_features)
        text_features = text_features + x

        return text_features


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_dropout_prob=0.1, proj_dropout_prob=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "Input size must be divisible by number of heads"

        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.final_projection = nn.Linear(embed_dim, embed_dim)

        self.attention_drop = nn.Dropout(attention_dropout_prob)
        self.proj_drop = nn.Dropout(proj_dropout_prob)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, image_features, text_features):
        bs = image_features.size(0)
        # Calculate query, key, and value
        query = self.query_projection(text_features)  # query由b得到, key和value由a得到
        key = self.key_projection(image_features)
        value = self.value_projection(image_features)

        # Split heads [bs, patch_num, num_head, head_dim]->[bs, num_head, patch_num, head_dim]
        query = query.view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = self.softmax(attention_scores)

        # dropout and Apply attention to value
        attention_weights = self.attention_drop(attention_weights)
        output = torch.matmul(attention_weights, value)

        # Concatenate and project back and dropout
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.head_dim)
        output = self.final_projection(output)
        output = self.proj_drop(output)

        return output


class TextHead(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = torch.sum(x * self.weight, 2)
        if self.bias is not None:
            x = x + self.bias
        return x


