import io

import torch

from mmdet.models.necks.fpn import FPN
from mmdet.models.necks.fsta_fpn import CSTA, TAFE, FSTABlock, FSTAFPN


def test_disabled_block_is_exact_identity():
    block = FSTABlock(
        channels=16,
        enable_tafe=False,
        enable_csta=False,
        need_foreground_head=False)
    feature = torch.randn(2, 16, 11, 13)
    output, logits = block(feature)
    assert output is feature
    assert logits is None


def test_tafe_and_csta_handle_odd_non_square_shapes_and_backward():
    feature = torch.randn(2, 32, 17, 23, requires_grad=True)
    valid = torch.ones(2, 1, 17, 23, dtype=torch.bool)
    valid[1, :, 15:, :] = False
    valid[1, :, :, 21:] = False
    foreground = torch.rand(2, 1, 17, 23)

    tafe = TAFE(32, use_foreground_gate=True)
    csta = CSTA(
        32,
        inner_dim=16,
        window_size=4,
        group_size=2,
        use_foreground=True)
    enhanced, _ = tafe(feature, foreground, valid)
    delta = csta(enhanced, foreground, valid)
    output = enhanced + delta

    assert output.shape == feature.shape
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(delta[1, :, 15:, :]) == 0
    output.square().mean().backward()
    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()
    assert any(parameter.grad is not None for parameter in csta.parameters())
    gradient_norm = sum(
        parameter.grad.float().norm() for parameter in csta.parameters()
        if parameter.grad is not None)
    assert torch.isfinite(gradient_norm) and gradient_norm > 0


def test_csta_tiny_input_and_partition_reverse_are_exact():
    csta = CSTA(
        4,
        inner_dim=4,
        window_size=3,
        group_size=3,
        use_foreground=True)
    feature = torch.arange(1 * 4 * 3 * 5, dtype=torch.float32).view(
        1, 4, 3, 5)
    valid = torch.ones(1, 1, 3, 5, dtype=torch.bool)
    foreground = torch.rand(1, 1, 3, 5)
    grouped, _, _, metadata = csta._partition(
        feature, valid, foreground)
    restored = csta._reverse(grouped, metadata)
    torch.testing.assert_close(restored, feature, rtol=0, atol=0)

    delta = csta(feature, foreground, valid)
    assert delta.shape == feature.shape
    assert torch.isfinite(delta).all()

    ids = torch.tensor([[2, -1, 0, 2, 1, 0]])
    order = torch.argsort(ids, dim=1)
    inverse = torch.argsort(order, dim=1)
    sorted_ids = torch.gather(ids, 1, order)
    restored_ids = torch.gather(sorted_ids, 1, inverse)
    torch.testing.assert_close(restored_ids, ids, rtol=0, atol=0)


def test_added_modules_ignore_values_in_invalid_padding():
    torch.manual_seed(3)
    block = FSTABlock(
        channels=8,
        enable_tafe=True,
        enable_csta=True,
        inner_dim=4,
        window_size=2,
        group_size=2,
        need_foreground_head=True).eval()
    valid = torch.zeros(1, 1, 9, 11, dtype=torch.bool)
    valid[:, :, :7, :8] = True
    first = torch.randn(1, 8, 9, 11)
    second = first.clone()
    second.masked_fill_(~valid.expand_as(second), 1000.)
    with torch.no_grad():
        first_out, first_logits = block(first, valid)
        second_out, second_logits = block(second, valid)
    valid_features = valid.expand_as(first_out)
    torch.testing.assert_close(
        first_out[valid_features], second_out[valid_features], rtol=0, atol=0)
    torch.testing.assert_close(
        first_logits[valid], second_logits[valid], rtol=0, atol=0)


def test_disabled_texture_gate_has_no_unused_gate_parameters():
    tafe = TAFE(8, use_foreground_gate=False, use_texture_gate=False)
    assert tafe.texture_gate is None
    assert all('texture_gate' not in name for name, _ in tafe.named_parameters())


def test_csta_window_one_and_all_mask_are_finite():
    csta = CSTA(4, inner_dim=2, window_size=1, group_size=2)
    feature = torch.randn(1, 4, 3, 5, requires_grad=True)
    valid = torch.zeros(1, 1, 3, 5, dtype=torch.bool)
    delta = csta(feature, valid_mask=valid)
    assert torch.count_nonzero(delta) == 0
    assert torch.isfinite(delta).all()
    delta.sum().backward()
    assert feature.grad is not None


def test_csta_discrete_foreground_assignment_has_no_continuous_gate_gradient():
    block = FSTABlock(
        channels=8,
        enable_tafe=False,
        enable_csta=True,
        csta_use_foreground=True,
        csta_foreground_bias=False,
        inner_dim=4,
        window_size=2,
        group_size=2,
        need_foreground_head=True)
    feature = torch.randn(1, 8, 7, 9, requires_grad=True)
    output, logits = block(feature)
    output.sum().backward(retain_graph=True)
    foreground_parameters = list(block.foreground_head.parameters())
    assert all(parameter.grad is None for parameter in foreground_parameters)
    logits.square().mean().backward()
    assert all(parameter.grad is not None for parameter in foreground_parameters)


def test_shared_block_processes_both_enhanced_levels():
    neck = FSTAFPN(
        in_channels=[8, 16, 32, 64],
        out_channels=8,
        start_level=0,
        num_outs=4,
        add_extra_convs=False,
        enhance_levels=(0, 1),
        feature_strides=(4, 8, 16, 32),
        shared_weights=True,
        enable_tafe=True,
        enable_csta=False,
        loss_fg_weight=.1)
    inputs = (
        torch.randn(1, 8, 17, 23),
        torch.randn(1, 16, 9, 12),
        torch.randn(1, 32, 5, 6),
        torch.randn(1, 64, 3, 3),
    )
    outputs, auxiliary = neck(inputs, return_aux=True)
    assert len(outputs) == 4
    assert set(auxiliary['foreground_logits']) == {0, 1}
    assert neck.blocks is None and neck.shared_block is not None


def test_fsta_fpn_produces_true_p2_to_p5_shapes():
    neck = FSTAFPN(
        in_channels=[32, 64, 128, 256],
        out_channels=16,
        start_level=0,
        num_outs=4,
        add_extra_convs=False,
        enhance_levels=(0, 1),
        feature_strides=(4, 8, 16, 32),
        enable_tafe=False,
        enable_csta=False,
        loss_fg_weight=0.0)
    inputs = (
        torch.randn(2, 32, 33, 49),
        torch.randn(2, 64, 17, 25),
        torch.randn(2, 128, 9, 13),
        torch.randn(2, 256, 5, 7),
    )
    outputs, auxiliary = neck(
        inputs,
        img_metas=[
            dict(img_shape=(127, 191, 3)),
            dict(img_shape=(93, 141, 3)),
        ],
        return_aux=True)
    assert [item.shape[-2:] for item in outputs] == [
        (33, 49), (17, 25), (9, 13), (5, 7)
    ]
    assert auxiliary['foreground_logits'] == {}
    assert auxiliary['valid_masks'][0].dtype == torch.bool
    assert auxiliary['valid_masks'][0][0].sum() > auxiliary[
        'valid_masks'][0][1].sum()


def test_disabled_fsta_fpn_matches_standard_fpn_exactly():
    kwargs = dict(
        in_channels=[32, 64, 128, 256],
        out_channels=16,
        start_level=0,
        num_outs=4,
        add_extra_convs=False)
    standard = FPN(**kwargs).eval()
    fsta = FSTAFPN(
        **kwargs,
        enhance_levels=(0, 1),
        feature_strides=(4, 8, 16, 32),
        enable_tafe=False,
        enable_csta=False,
        loss_fg_weight=0.0).eval()
    missing, unexpected = fsta.load_state_dict(
        standard.state_dict(), strict=False)
    assert not unexpected
    assert not missing
    inputs = (
        torch.randn(1, 32, 31, 47),
        torch.randn(1, 64, 16, 24),
        torch.randn(1, 128, 8, 12),
        torch.randn(1, 256, 4, 6),
    )
    with torch.no_grad():
        expected = standard(inputs)
        actual = fsta(inputs)
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(
            actual_level, expected_level, rtol=0, atol=0)


def test_block_state_dict_roundtrip_is_exact():
    torch.manual_seed(9)
    block = FSTABlock(
        channels=16,
        enable_tafe=True,
        enable_csta=True,
        inner_dim=8,
        window_size=4,
        group_size=2,
        need_foreground_head=True).eval()
    feature = torch.randn(1, 16, 13, 19)
    valid = torch.ones(1, 1, 13, 19, dtype=torch.bool)
    with torch.no_grad():
        expected = block(feature, valid)[0]
    buffer = io.BytesIO()
    torch.save(block.state_dict(), buffer)
    buffer.seek(0)

    restored = FSTABlock(
        channels=16,
        enable_tafe=True,
        enable_csta=True,
        inner_dim=8,
        window_size=4,
        group_size=2,
        need_foreground_head=True).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        actual = restored(feature, valid)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_foreground_initialization_is_idempotent_and_configurable():
    neck = FSTAFPN(
        in_channels=[8, 16, 32, 64],
        out_channels=8,
        start_level=0,
        num_outs=4,
        add_extra_convs=False,
        enhance_levels=(0, 1),
        feature_strides=(4, 8, 16, 32),
        enable_tafe=True,
        enable_csta=False,
        foreground_prior_prob=0.02)
    neck.init_weights()
    expected = {
        key: value.detach().clone()
        for key, value in neck.state_dict().items()
        if 'foreground_head' in key
    }
    neck.init_weights()
    actual = {
        key: value.detach().clone()
        for key, value in neck.state_dict().items()
        if 'foreground_head' in key
    }
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
