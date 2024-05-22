import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAlignmentLoss(nn.Module):
    def __init__(self, logit_scale, batch_size):
        super(GlobalAlignmentLoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale)
        self.batch_size = batch_size
        self.global_alignment_labels = None

    def forward(self, image_embed, table_embed):
        logit_scale = self.logit_scale.exp()

        local_batch_size = image_embed.size(0)


        if local_batch_size != self.batch_size:
            self.global_alignment_labels = torch.arange(local_batch_size, device=image_embed.device).long()
            self.batch_size = local_batch_size


        image_embed = F.normalize(image_embed, dim=-1, p=2)
        table_embed = F.normalize(table_embed, dim=-1, p=2)


        logits_per_image = logit_scale * torch.matmul(image_embed, table_embed.T)
        logits_per_table = logit_scale * torch.matmul(table_embed, image_embed.T)

        image_loss = F.cross_entropy(logits_per_image, self.global_alignment_labels)
        table_loss = F.cross_entropy(logits_per_table, self.global_alignment_labels)

        # 计算图像损失和表格损失的平均值
        loss = (image_loss + table_loss) / 2


        return loss



class LocalAlignmentLoss(nn.Module):
    def __init__(self, local_cross_attention, predictor, logit_scale):
        super(LocalAlignmentLoss, self).__init__()
        self.local_cross_attention = local_cross_attention
        self.predictor = predictor
        self.local_logit_scale = nn.Parameter(torch.ones([]) * logit_scale)
        self.lc_labels = None

    def forward(self, local_image_embed_stacks, local_table_embed_stacks):
        total_image_loss = 0.0
        total_table_loss = 0.0
        table_to_local_image_embed_stacks = []

        for idx in range(local_image_embed_stacks.size(0)):
            local_table_embed = local_table_embed_stacks[idx]
            local_image_embed = local_image_embed_stacks[idx]
            # 使用本地交叉注意力模块进行特征对齐
            table_to_local_image_embed, _, image_to_local_table_embed, _ = self.local_cross_attention(local_image_embed,
                                                                                                      local_table_embed)
            # 计算SimSiam损失
            image_loss = self.simsiam_loss_func(local_image_embed, table_to_local_image_embed)
            # 计算表格的本地对齐损失
            table_loss = self.table_local_loss_fn(local_table_embed, image_to_local_table_embed)
            total_image_loss += image_loss  # 累积图像损失
            total_table_loss += table_loss  # 累积表格损失
            # 将表格到图像的嵌入添加到堆栈中
            table_to_local_image_embed_stacks.append(table_to_local_image_embed.unsqueeze(0))

        # 将嵌入堆栈合并
        self.table_to_local_image_embed_stacks = torch.cat(table_to_local_image_embed_stacks, dim=0)

        return total_image_loss / local_image_embed_stacks.size(0), total_table_loss / local_image_embed_stacks.size(0)

    def table_local_loss_fn(self, embed_A, embed_B, norm=True):
        # 计算logit比例的指数值，将对数标度转换为线性标度
        logit_scale = self.local_logit_scale.exp()
        if norm:
            embed_A = F.normalize(embed_A, dim=-1, p=2)
            embed_B = F.normalize(embed_B, dim=-1, p=2)
        # 生成本地对齐标签，范围为当前批次大小
        self.lc_labels = torch.arange(embed_B.size(0), device=embed_B.device).long()
        logits_per_image = logit_scale * embed_B @ embed_A.T
        logits_per_table = logit_scale * embed_A @ embed_B.T
        image_loss = F.cross_entropy(logits_per_image, self.lc_labels)
        table_loss = F.cross_entropy(logits_per_table, self.lc_labels)

        return (image_loss + table_loss) / 2

    def simsiam_loss_func(self, x, y):
        # 使用预测器生成预测嵌入
        p_x = self.predictor(x)
        p_y = self.predictor(y)
        # 计算余弦相似度并返回损失
        z_x = x.detach()
        z_y = y.detach()
        return - (F.cosine_similarity(p_x, z_y, dim=-1).mean() + F.cosine_similarity(p_y, z_x, dim=-1).mean()) * 0.5




    class NT_XentLoss(torch.nn.Module):
        def __init__(self, hparams, device='cuda' if torch.cuda.is_available() else 'cpu'):
            super(NT_XentLoss, self).__init__()
            self.batch_size = hparams.batch_size
            self.temperature = hparams.temperature
            self.device = device
            self.softmax = torch.nn.Softmax(dim=-1)

        def forward(self, z_i, z_j):
            # 计算两组嵌入间的点积相似度矩阵
            sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature

            # 对角线上的元素是自身与自身的相似度，应设置为非常小的负数，以避免自我对比
            mask = torch.eye(self.batch_size, device=self.device).bool()
            sim_matrix = sim_matrix.masked_fill(mask, value=float('-inf'))

            # 计算softmax以获得每个样本与其他样本的相似度概率分布
            sim_probs = self.softmax(sim_matrix)

            # 只取上三角或下三角的部分，因为矩阵是对称的，且已排除对角线
            sim_probs = sim_probs.triu(diagonal=1)

            # 计算每行的损失（除了自己以外的所有负对的log似然）
            loss_per_row = -torch.log(sim_probs.sum(dim=1))

            # 平均所有样本的损失
            loss = loss_per_row.mean()
            return loss