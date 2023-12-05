import os
import pdb
import torch
import torch.nn as nn
from pytorch_metric_learning.losses import NTXentLoss
from transformers import BertForMaskedLM, BertForPreTraining, BertTokenizer
from transformers import BertTokenizer, BertModel
import numpy as np


def mask_tokens(inputs, tokenizer, not_mask_pos=None):
    """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

    Args:
        inputs: Inputs to mask. (batch_size, max_length)
        tokenizer: Tokenizer.
        not_mask_pos: Using to forbid masking entity mentions. 1 for not mask.

    Returns:
        inputs: Masked inputs.
        labels: Masked language model labels.
    """
    # 检查tokenizer是否有mask token，没有会引发' ValueError '
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    # The `labels` tensor is created by cloning the `inputs` tensor. Serve as ground truth
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    # 创建一个与“标签”形状相同的概率矩阵，每个元素的概率为0.15。
    probability_matrix = torch.full(labels.shape, 0.15)

    # 处理特殊tokens，概率设置为0
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]  # eg.[CLS][SEP]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

    # 处理padding tokens
    # 如果tokenizer has a padding token，也被设置为0
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    # 使用伯努利分布，选择随机指标进行屏蔽。如果提供了' not_mask_pos '，这些位置将被排除在屏蔽过程之外。
    if not_mask_pos is None:
        masked_indices = torch.bernoulli(probability_matrix).bool()
    else:
        masked_indices = torch.bernoulli(probability_matrix).bool() & (
            ~(not_mask_pos.bool()))  # ** can't mask entity marker **
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # mask
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs.cuda(), labels.cuda()


class one_hot_CrossEntropy(torch.nn.Module):
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        super(one_hot_CrossEntropy, self).__init__()

    def forward(self, x, y):
        P_i = torch.nn.functional.softmax(x, dim=1)
        loss = y * torch.log(P_i + 0.0000001)
        loss = - torch.mean(torch.sum(loss, dim=1), dim=0)
        return loss


class CP(nn.Module):
    """Contrastive Pre-training model.

    This class implements `CP` model based on model `BertForMaskedLM`. And we
    use NTXentLoss as contrastive loss function.

    Attributes:
        model: Model to train.
        tokenizer: Tokenizer.
        ntxloss: Contrastive loss function.
        args: Args from command line.
    """

    def __init__(self, args):
        super(CP, self).__init__()
        # Initialize BERT models for masked language modeling
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.model2 = BertForMaskedLM.from_pretrained('bert-base-uncased')
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Initialize contrastive loss function and other parameters
        self.ntxloss = NTXentLoss(temperature=args.temperature)
        self.onehotloss1 = one_hot_CrossEntropy()
        self.onehotloss2 = one_hot_CrossEntropy()
        self.args = args
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, input, mask, rel_input, rel_mask, h_pos, t_pos):
        input = input.type(torch.long)
        h_pos = h_pos.type(torch.long)
        t_pos = t_pos.type(torch.long)

        #######################################
        rel_input_np = rel_input.numpy()
        sorted_idx = np.lexsort(tuple(rel_input_np.T))
        new_rel = rel_input[sorted_idx]
        new_input = input[sorted_idx]
        size = rel_input.shape[0]
        pos = torch.zeros(size, size)

        _, indices = torch.unique(new_input, dim=0, return_inverse=True)
        counts = torch.bincount(indices)
        counts_list = counts.int().tolist()

        current_row = 0
        for count in counts_list:
            pos[current_row:current_row + count, current_row:current_row + count] = 1
            current_row += count

        rel_mask = pos
        #################################
        # Ensure that `mask_tokens` function doesn't mask entity mention.
        indice = torch.arange(0, input.size()[0])
        not_mask_pos = torch.zeros((input.size()[0], input.size()[1]), dtype=int)
        not_mask_pos[indice, h_pos] = 1
        not_mask_pos[indice, t_pos] = 1

        # 将input通过mask_tokens()生成对应的 mask_input 和 mask_labels
        m_input, m_labels = mask_tokens(input.cpu(), self.tokenizer, not_mask_pos)
        m_rel_input, m_rel_labels = mask_tokens(rel_input.cpu(), self.tokenizer)
        # Masked Language Model Loss
        # 放进BERT模型中，提取出outputs和m_loss
        m_outputs = self.model(input_ids=m_input, labels=m_labels, attention_mask=mask, output_hidden_states=True)
        outputs = m_outputs[2][-1]
        # rel_outputs = self.model2(rel_input, attention_mask=rel_mask)
        m_loss = m_outputs[0]

        # Relation Prediction Loss
        # 一样的，m_rel 放进BERT提取
        m_rel_outputs = self.model2(input_ids=m_rel_input, labels=m_rel_labels, attention_mask=rel_mask,
                                    output_hidden_states=True)
        m_rel_loss = m_rel_outputs[0]
        rel_outputs = m_rel_outputs[2][-1]

        ####################################################
        # entity marker starter
        batch_size = input.size()[0]
        batch_rel = rel_mask.shape[0]
        indice = torch.arange(0, batch_size)
        rel_indice = torch.arange(0, batch_rel)
        h_state = outputs[indice, h_pos]  # (batch_size * 2, hidden_size) #head hidden state
        t_state = outputs[indice, t_pos]  # tail hidden state

        # r_state_global = rel_outputs[1]
        # r_state_local = torch.mean(rel_outputs[0], 1)
        cls_pos = torch.zeros(batch_rel).type(torch.long)  # index [CLS] position for each relation
        r_state_global = rel_outputs[rel_indice, cls_pos]
        r_state_local = torch.mean(rel_outputs, 1)  # 计算global的第一维度平均值->本地
        r_state = torch.cat((r_state_global, r_state_local), 1)  # 在二维连接global和local
        e_state = torch.cat((h_state, t_state), 1)
        # state = e_state*r_state

        # r_loss = self.ntxloss(state, label)

        r_state = r_state / r_state.norm(dim=1, keepdim=True)
        e_state = e_state / e_state.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_entity = logit_scale * e_state @ r_state.t()
        logits_per_relation = logits_per_entity.t()

        # batch_size = logits_per_entity.shape[0]

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # labels = torch.arange(batch_size, device=_device).long()
        final_pos = final_pos.to(_device)
        ####################################################

        loss_entity = self.onehotloss1(logits_per_entity, final_pos)
        loss_relation = self.onehotloss2(logits_per_relation, final_pos.t())

        r_loss = (loss_entity + loss_relation) / 2
        return m_loss + m_rel_loss, r_loss


class MTB(nn.Module):
    """Matching the Blanks.

    This class implements `MTB` model based on model `BertForMaskedLM`.

    Attributes:
        model: Model to train.
        tokenizer: Tokenizer.
        bceloss: Binary Cross Entropy loss.
    """

    def __init__(self, args):
        super(MTB, self).__init__()
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bceloss = nn.BCEWithLogitsLoss()
        self.args = args

    def forward(self, l_input, l_mask, l_ph, l_pt, r_input, r_mask, r_ph, r_pt, label):
        # compute not mask entity marker
        indice = torch.arange(0, l_input.size()[0])
        l_not_mask_pos = torch.zeros((l_input.size()[0], l_input.size()[1]), dtype=int)
        r_not_mask_pos = torch.zeros((l_input.size()[0], l_input.size()[1]), dtype=int)

        # Ensure that `mask_tokens` function doesn't mask entity mention.
        l_not_mask_pos[indice, l_ph] = 1
        l_not_mask_pos[indice, l_pt] = 1

        r_not_mask_pos[indice, r_ph] = 1
        r_not_mask_pos[indice, r_pt] = 1

        # masked language model loss
        m_l_input, m_l_labels = mask_tokens(l_input.cpu(), self.tokenizer, l_not_mask_pos)
        m_r_input, m_r_labels = mask_tokens(r_input.cpu(), self.tokenizer, r_not_mask_pos)
        m_l_outputs = self.model(input_ids=m_l_input, labels=m_l_labels, attention_mask=l_mask)
        m_r_outputs = self.model(input_ids=m_r_input, labels=m_r_labels, attention_mask=r_mask)
        m_loss = m_l_outputs[1] + m_r_outputs[1]

        # sentence pair relation loss
        l_outputs = m_l_outputs
        r_outputs = m_r_outputs

        batch_size = l_input.size()[0]
        indice = torch.arange(0, batch_size)

        # left output
        l_h_state = l_outputs[0][indice, l_ph]  # (batch, hidden_size)
        l_t_state = l_outputs[0][indice, l_pt]  # (batch, hidden_size)
        l_state = torch.cat((l_h_state, l_t_state), 1)  # (batch, 2 * hidden_size)

        # right output
        r_h_state = r_outputs[0][indice, r_ph]
        r_t_state = r_outputs[0][indice, r_pt]
        r_state = torch.cat((r_h_state, r_t_state), 1)

        # cal similarity
        similarity = torch.sum(l_state * r_state, 1)  # (batch)

        # cal loss
        r_loss = self.bceloss(similarity, label.float())

        return m_loss, r_loss



