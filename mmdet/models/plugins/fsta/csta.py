"""Bounded similarity aggregation. Adapted from CUST; see third_party/."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSTA(nn.Module):
    """Cross-window Similarity-guided Target Aggregation.

    Attention is restricted to a bounded search region of
    ``group_size x group_size`` windows.  Tokens are assigned to the most
    similar window representative, sorted, processed in fixed-size chunks and
    restored to their original spatial positions.
    """

    def __init__(self,
                 channels,
                 inner_dim=64,
                 window_size=8,
                 group_size=4,
                 use_foreground=False,
                 foreground_bias=False,
                 foreground_bias_weight=1.0,
                 eps=1e-6):
        super().__init__()
        if window_size < 1 or group_size < 1:
            raise ValueError('window_size and group_size must be positive')
        self.channels = channels
        self.inner_dim = inner_dim
        self.window_size = window_size
        self.group_size = group_size
        self.use_foreground = use_foreground
        self.foreground_bias = foreground_bias
        self.foreground_bias_weight = foreground_bias_weight
        self.eps = eps
        self.scale = inner_dim**-0.5

        self.input_proj = nn.Conv2d(channels, inner_dim, 1, bias=False)
        self.to_q = nn.Linear(inner_dim, inner_dim)
        self.to_k = nn.Linear(inner_dim, inner_dim)
        self.to_v = nn.Linear(inner_dim, inner_dim)
        self.gate_proj = nn.Linear(inner_dim, inner_dim)
        self.token_proj = nn.Linear(inner_dim, inner_dim)
        self.output_proj = nn.Conv2d(inner_dim, channels, 1, bias=False)

    def _partition(self, x, valid_mask, foreground_prob):
        batch, channels, height, width = x.shape
        ws, gs = self.window_size, self.group_size
        region = ws * gs
        pad_h = (region - height % region) % region
        pad_w = (region - width % region) % region

        x = F.pad(x, (0, pad_w, 0, pad_h), value=0)
        valid_mask = F.pad(
            valid_mask.to(dtype=torch.bool), (0, pad_w, 0, pad_h), value=False)
        if foreground_prob is None:
            foreground_prob = x.new_ones((batch, 1, height, width))
        foreground_prob = F.pad(
            foreground_prob, (0, pad_w, 0, pad_h), value=0)

        padded_h, padded_w = x.shape[-2:]
        groups_h, groups_w = padded_h // region, padded_w // region

        x = x.view(batch, channels, groups_h, gs, ws, groups_w, gs,
                   ws).permute(0, 2, 5, 3, 6, 4, 7, 1).contiguous()
        x = x.view(batch, groups_h * groups_w, gs * gs, ws * ws,
                   channels)

        def partition_map(value):
            value = value.view(batch, 1, groups_h, gs, ws, groups_w, gs,
                               ws)
            value = value.permute(0, 2, 5, 3, 6, 4, 7, 1).contiguous()
            return value.view(batch, groups_h * groups_w, gs * gs,
                              ws * ws, 1)

        valid_mask = partition_map(valid_mask)
        foreground_prob = partition_map(foreground_prob)
        metadata = (height, width, padded_h, padded_w, groups_h, groups_w)
        return x, valid_mask, foreground_prob, metadata

    def _reverse(self, x, metadata):
        height, width, padded_h, padded_w, groups_h, groups_w = metadata
        batch, _, _, _, channels = x.shape
        ws, gs = self.window_size, self.group_size
        x = x.view(batch, groups_h, groups_w, gs, gs, ws, ws,
                   channels).permute(0, 7, 1, 3, 5, 2, 4, 6).contiguous()
        x = x.view(batch, channels, padded_h, padded_w)
        return x[:, :, :height, :width]

    @staticmethod
    def _neighbor_kv(chunks, pad_value=0):
        """Build previous-half/current/next-half key-value chunks."""
        batch_groups, num_chunks, chunk_size = chunks.shape[:3]
        neighbor_size = chunk_size // 2
        if neighbor_size == 0:
            return chunks
        tail_shape = list(chunks.shape)
        tail_shape[1] = 1
        tail_shape[2] = neighbor_size
        pad = chunks.new_full(tail_shape, pad_value)
        previous = torch.cat((pad, chunks[:, :-1, -neighbor_size:]), dim=1)
        following = torch.cat((chunks[:, 1:, :neighbor_size], pad), dim=1)
        return torch.cat((previous, chunks, following), dim=2)

    def forward(self, x, foreground_prob=None, valid_mask=None):
        if valid_mask is None:
            valid_mask = torch.ones(
                (x.size(0), 1, x.size(2), x.size(3)),
                device=x.device,
                dtype=torch.bool)
        if foreground_prob is not None:
            foreground_prob = foreground_prob.to(dtype=x.dtype)

        original_valid = valid_mask
        projected = self.input_proj(x * valid_mask.to(dtype=x.dtype))
        grouped, valid, foreground, metadata = self._partition(
            projected, valid_mask, foreground_prob)
        batch, num_groups, num_windows, num_tokens, dim = grouped.shape

        # Assignment is intentionally discrete, as in the reference method.
        # Gradients flow through the subsequent attention and gates, not through
        # argmax/argsort indices.
        with torch.no_grad():
            representative_weight = valid.to(dtype=grouped.dtype)
            if self.use_foreground:
                representative_weight = representative_weight * (
                    self.eps + foreground)
            denominator = representative_weight.sum(dim=3).clamp_min(self.eps)
            representatives = (
                grouped * representative_weight).sum(dim=3) / denominator
            representative_valid = valid.any(dim=3).squeeze(-1)
            token_norm = F.normalize(grouped.float(), dim=-1)
            representative_norm = F.normalize(
                representatives.float(), dim=-1)
            similarity = torch.einsum('bgwpd,bgkd->bgwpk', token_norm,
                                      representative_norm)
            similarity = similarity.masked_fill(
                ~representative_valid[:, :, None, None, :], -1e4)
            assignment = similarity.argmax(dim=-1)
            assignment = assignment.masked_fill(~valid.squeeze(-1), -1)

        flat_count = num_windows * num_tokens
        batch_groups = batch * num_groups
        flat_x = grouped.view(batch_groups, flat_count, dim)
        flat_ids = assignment.view(batch_groups, flat_count)
        flat_valid = valid.view(batch_groups, flat_count)
        flat_foreground = foreground.view(batch_groups, flat_count)

        sorting_indices = torch.argsort(flat_ids, dim=1)
        gather_features = sorting_indices.unsqueeze(-1).expand(-1, -1, dim)
        sorted_x = torch.gather(flat_x, 1, gather_features)
        sorted_ids = torch.gather(flat_ids, 1, sorting_indices)
        sorted_valid = torch.gather(flat_valid, 1, sorting_indices)
        sorted_foreground = torch.gather(flat_foreground, 1, sorting_indices)

        chunk_size = num_tokens
        x_chunks = sorted_x.view(batch_groups, num_windows, chunk_size, dim)
        id_chunks = sorted_ids.view(batch_groups, num_windows, chunk_size)
        valid_chunks = sorted_valid.view(batch_groups, num_windows, chunk_size)
        fg_chunks = sorted_foreground.view(batch_groups, num_windows,
                                           chunk_size)

        kv_x = self._neighbor_kv(x_chunks, pad_value=0)
        kv_ids = self._neighbor_kv(id_chunks, pad_value=-2)
        kv_valid = self._neighbor_kv(valid_chunks, pad_value=False)
        kv_foreground = self._neighbor_kv(fg_chunks, pad_value=0)

        query = self.to_q(x_chunks)
        key = self.to_k(kv_x)
        value = self.to_v(kv_x)
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if self.foreground_bias and self.use_foreground:
            bias = torch.log(kv_foreground.clamp_min(self.eps))
            logits = logits + self.foreground_bias_weight * bias.unsqueeze(-2)

        attention_mask = (
            (id_chunks.unsqueeze(-1) == kv_ids.unsqueeze(-2))
            & valid_chunks.unsqueeze(-1) & kv_valid.unsqueeze(-2)
            & (id_chunks.unsqueeze(-1) >= 0))
        has_key = attention_mask.any(dim=-1, keepdim=True)
        safe_mask = attention_mask.clone()
        safe_mask[..., 0] = safe_mask[..., 0] | ~has_key.squeeze(-1)
        logits = logits.masked_fill(~safe_mask,
                                    torch.finfo(logits.dtype).min)
        attention = F.softmax(logits, dim=-1)
        attention = attention * attention_mask.to(dtype=attention.dtype)
        attention = attention / attention.sum(
            dim=-1, keepdim=True).clamp_min(self.eps)

        out = torch.matmul(attention, value)
        out = out * torch.sigmoid(self.gate_proj(x_chunks))
        out = out * valid_chunks.unsqueeze(-1).to(dtype=out.dtype)
        out = self.token_proj(out).reshape(batch_groups, flat_count, dim)

        # Build the inverse permutation in O(N) instead of sorting twice.
        inverse_indices = torch.empty_like(sorting_indices)
        positions = torch.arange(
            flat_count, device=sorting_indices.device).view(1, -1)
        positions = positions.expand_as(sorting_indices)
        inverse_indices.scatter_(1, sorting_indices, positions)
        inverse_features = inverse_indices.unsqueeze(-1).expand(-1, -1, dim)
        out = torch.gather(out, 1, inverse_features)
        out = out.view(batch, num_groups, num_windows, num_tokens, dim)
        out = self._reverse(out, metadata)
        out = self.output_proj(out)
        return out * original_valid.to(dtype=out.dtype)
