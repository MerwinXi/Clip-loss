import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAlignmentLoss(nn.Module):
    def __init__(self, logit_scale, batch_size):
        super(GlobalAlignmentLoss, self).__init__()
        # 初始化对比损失的logit比例参数，并将其转换为nn.Parameter以便进行反向传播
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale)
        self.batch_size = batch_size
        self.global_alignment_labels = None

    def forward(self, image_embed, text_embed):
        # 计算logit比例的指数值，将对数标度转换为线性标度
        logit_scale = self.logit_scale.exp()
        
        
        local_batch_size = image_embed.size(0)
        
        # 如果当前批次大小与之前的不同，则更新全局对齐标签
        if local_batch_size != self.batch_size:
            # 生成全局对齐标签，范围为当前批次大小
            self.global_alignment_labels = torch.arange(local_batch_size, device=image_embed.device).long()
            # 更新批次大小
            self.batch_size = local_batch_size

        # 对图像嵌入进行L2归一化，使其值在单位球面上
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        # 对文本嵌入进行L2归一化，使其值在单位球面上
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # 计算图像嵌入和文本嵌入之间的相似度矩阵
        logits_per_image = logit_scale * torch.matmul(image_embed, text_embed.T)
        logits_per_text = logit_scale * torch.matmul(text_embed, image_embed.T)
        
        # 计算图像嵌入的交叉熵损失
        image_loss = F.cross_entropy(logits_per_image, self.global_alignment_labels)
        # 计算文本嵌入的交叉熵损失
        text_loss = F.cross_entropy(logits_per_text, self.global_alignment_labels)
        
        # 计算图像损失和文本损失的平均值
        loss = (image_loss + text_loss) / 2
        
        # 返回最终的全局对齐损失
        return loss

class LocalAlignmentLoss(nn.Module):
    def __init__(self, local_cross_attention, predictor, logit_scale):
        super(LocalAlignmentLoss, self).__init__()
        # 本地交叉注意力模块，用于提取本地特征
        self.local_cross_attention = local_cross_attention
        # 预测器模块，用于生成预测嵌入
        self.predictor = predictor
        # 初始化对比损失的logit比例参数，并将其转换为nn.Parameter以便进行反向传播
        self.local_logit_scale = nn.Parameter(torch.ones([]) * logit_scale)
        self.lc_labels = None  # 初始化标签

    def forward(self, local_image_embed_stacks, local_text_embed_stacks):
        total_image_loss = 0.0  # 初始化图像总损失
        total_text_loss = 0.0  # 初始化文本总损失
        text_to_local_image_embed_stacks = []  # 初始化文本到图像的嵌入堆栈

        for idx in range(local_image_embed_stacks.size(0)):
            local_text_embed = local_text_embed_stacks[idx]
            local_image_embed = local_image_embed_stacks[idx]
            # 使用本地交叉注意力模块进行特征对齐
            text_to_local_image_embed, _, image_to_local_text_embed, _ = self.local_cross_attention(local_image_embed, local_text_embed)
            # 计算SimSiam损失
            image_loss = self.simsiam_loss_func(local_image_embed, text_to_local_image_embed)
            # 计算文本的本地对齐损失
            text_loss = self.text_local_loss_fn(local_text_embed, image_to_local_text_embed)
            total_image_loss += image_loss  # 累积图像损失
            total_text_loss += text_loss  # 累积文本损失
            # 将文本到图像的嵌入添加到堆栈中
            text_to_local_image_embed_stacks.append(text_to_local_image_embed.unsqueeze(0))

        # 将嵌入堆栈合并
        self.text_to_local_image_embed_stacks = torch.cat(text_to_local_image_embed_stacks, dim=0)
        # 返回平均损失
        return total_image_loss / local_image_embed_stacks.size(0), total_text_loss / local_image_embed_stacks.size(0)

    def text_local_loss_fn(self, embed_A, embed_B, norm=True):
        # 计算logit比例的指数值，将对数标度转换为线性标度
        logit_scale = self.local_logit_scale.exp()
        if norm:
            # 对嵌入进行L2归一化
            embed_A = F.normalize(embed_A, dim=-1, p=2)
            embed_B = F.normalize(embed_B, dim=-1, p=2)
        # 生成本地对齐标签，范围为当前批次大小
        self.lc_labels = torch.arange(embed_B.size(0), device=embed_B.device).long()
        # 计算图像和文本嵌入之间的相似度矩阵
        logits_per_image = logit_scale * embed_B @ embed_A.T
        logits_per_text = logit_scale * embed_A @ embed_B.T
        # 计算图像嵌入的交叉熵损失
        image_loss = F.cross_entropy(logits_per_image, self.lc_labels)
        # 计算文本嵌入的交叉熵损失
        text_loss = F.cross_entropy(logits_per_text, self.lc_labels)
        # 返回图像损失和文本损失的平均值
        return (image_loss + text_loss) / 2

    def simsiam_loss_func(self, x, y):
        # 使用预测器生成预测嵌入
        p_x = self.predictor(x)
        p_y = self.predictor(y)
        # 计算余弦相似度并返回损失
        z_x = x.detach()
        z_y = y.detach()
        return - (F.cosine_similarity(p_x, z_y, dim=-1).mean() + F.cosine_similarity(p_y, z_x, dim=-1).mean()) * 0.5
