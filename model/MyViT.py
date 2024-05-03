import torch.nn as nn
import timm


class MyViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=False,
                 checkpoint_path='../parameters/ViT_model/vit_base_patch12_224_augreg2_in21k_ft_in1k.bin',
                 num_classes=20):
        super(MyViT, self).__init__()
        # 使用checkpoint加载预训练的模型
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained,
                                  checkpoint_path=checkpoint_path)
        # 修改最终的全连接层
        self.model.head = nn.Linear(self.model.head.in_features, 20)

    def forward(self, x):
        x = self.model(x)

        return x  # 包括了每一个大的block的输出
