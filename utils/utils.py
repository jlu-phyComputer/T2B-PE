import torch.nn as nn
import torch
import torch.nn.functional as F


def conv_3x3x3(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)


def compute_tensor_cos_sim(tensor):
    batch_size, token_count, channel_count = tensor.size()

    cosine_similarities_batchwise = torch.zeros(batch_size, token_count, token_count)

    for i in range(batch_size):
        current_batch_tensor = tensor[i]

        reshaped_tensor = current_batch_tensor.view(token_count, channel_count)

        cosine_similarities = F.cosine_similarity(reshaped_tensor.unsqueeze(1), reshaped_tensor.unsqueeze(0), dim=2)

        cosine_similarities = F.softmax(cosine_similarities, dim=-1)

        cosine_similarities_batchwise[i] = cosine_similarities

    return cosine_similarities_batchwise
