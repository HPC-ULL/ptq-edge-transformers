"""BatchNorm folding (quantization/vit_brevitas/bn_fold.py) on small modules
shaped like timm's ConvNormAct containers. Needs torch (skipped without it)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from quantization.vit_brevitas import bn_fold  # noqa: E402


def _randomize_bn(bn):
    with torch.no_grad():
        bn.running_mean.uniform_(-2, 2)
        bn.running_var.uniform_(0.1, 4.0)
        if bn.affine:
            bn.weight.uniform_(0.5, 2.0)
            bn.bias.uniform_(-1, 1)
    return bn


class BatchNormAct2d(nn.BatchNorm2d):
    """Like timm's: a BatchNorm2d subclass that also applies the activation
    (in-place, as timm creates it by default)."""

    def __init__(self, n):
        super().__init__(n)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(super().forward(x))


class ConvNormAct(nn.Module):
    """timm.layers style: .conv then .bn (BatchNormAct2d)."""

    def __init__(self, cin, cout, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, groups=groups, bias=bias)
        self.bn = _randomize_bn(BatchNormAct2d(cout))

    def forward(self, x):
        return self.bn(self.conv(x))


class EffVitConvNormAct(nn.Module):
    """efficientvit_mit style: .conv then .norm (plain BN), then .act."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, bias=False)
        self.norm = _randomize_bn(nn.BatchNorm2d(cout))
        self.act = nn.Hardswish()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


EffVitConvNormAct.__name__ = "ConvNormAct"  # timm's class name in efficientvit_mit


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = ConvNormAct(3, 8)
        self.b = ConvNormAct(8, 8, groups=8, bias=True)  # depthwise, conv with its own bias
        self.c = EffVitConvNormAct(8, 16)
        self.loose_bn = _randomize_bn(nn.BatchNorm2d(16))  # not inside a known container: must be left alone

    def forward(self, x):
        return self.loose_bn(self.c(self.b(self.a(x))))


def _net():
    torch.manual_seed(0)
    return Net().eval()


def test_folding_preserves_the_float_output_and_counts_pairs():
    net = _net()
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        before = net(x)
        folded, left, kept = bn_fold.fold_batchnorm(net)
        after = net(x)

    assert (folded, left, kept) == (3, 1, 3)
    assert torch.allclose(before, after, rtol=1e-5, atol=1e-5)


def test_fold_conv_bn_leaves_an_exact_identity_bn_and_gives_the_conv_a_bias():
    net = _net()
    bn = net.c.norm
    bn_fold.fold_conv_bn(net.c.conv, bn)

    assert torch.all(bn.running_mean == 0) and torch.all(bn.running_var == 1) and bn.eps == 0.0
    assert torch.all(bn.weight == 1) and torch.all(bn.bias == 0)
    x = torch.randn(1, 16, 4, 4)
    with torch.no_grad():
        assert torch.equal(bn(x), x)  # exactly x, so removing it cannot change anything
    assert isinstance(net.c.conv.bias, nn.Parameter)  # created (the conv had none)


def test_by_default_the_folded_bns_are_kept_as_exact_identities():
    net = _net()
    x = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        before = net(x)
        assert bn_fold.fold_batchnorm(net) == (3, 1, 3)
        after = net(x)

    assert type(net.a.bn).__name__ == "BatchNormAct2d" and isinstance(net.c.norm, nn.BatchNorm2d)
    assert torch.all(net.a.bn.running_var == 1) and net.a.bn.eps == 0.0
    assert sum(isinstance(m, nn.BatchNorm2d) for m in net.modules()) == 4
    assert torch.allclose(before, after, rtol=1e-5, atol=1e-5)


def test_remove_bn_replaces_the_folded_bns_by_what_they_do_after_normalizing():
    net = _net()
    assert bn_fold.fold_batchnorm(net, remove_bn=True) == (3, 1, 0)

    # timm BatchNormAct2d -> Sequential(drop, act): the activation survives
    assert isinstance(net.a.bn, nn.Sequential) and isinstance(net.a.bn[-1], nn.ReLU)
    assert isinstance(net.b.bn, nn.Sequential)
    assert isinstance(net.c.norm, nn.Identity)  # plain BN -> Identity (its act is a separate module)
    assert isinstance(net.loose_bn, nn.BatchNorm2d) and net.loose_bn.running_var.min() >= 0.1  # untouched
    assert sum(isinstance(m, nn.BatchNorm2d) for m in net.modules()) == 1


@pytest.mark.parametrize("remove_bn", [False, True])
def test_the_activation_inside_batchnormact_is_kept(remove_bn):
    net = _net()
    bn_fold.fold_batchnorm(net, remove_bn=remove_bn)
    with torch.no_grad():
        out = net.a(torch.randn(1, 3, 8, 8))
    assert out.min() >= 0  # ReLU still applied after folding


def test_checked_folding_reports_and_accepts_an_exact_fold(capsys):
    folded, left, kept, rel = bn_fold.fold_batchnorm_checked(_net(), (1, 3, 16, 16), label="net")

    assert (folded, left, kept) == (3, 1, 3) and rel < 1e-5
    assert "3 Conv+BN pair(s) folded, 0 folded BN removed, 3 kept as identity, 1 BatchNorm2d left unfolded" in \
        capsys.readouterr().out


def test_checked_folding_with_remove_bn_reports_the_removal(capsys):
    folded, left, kept, rel = bn_fold.fold_batchnorm_checked(_net(), (1, 3, 16, 16), label="net", remove_bn=True)

    assert (folded, left, kept) == (3, 1, 0) and rel < 1e-5
    assert "3 Conv+BN pair(s) folded, 3 folded BN removed, 0 kept as identity, 1 BatchNorm2d left unfolded" in \
        capsys.readouterr().out


def test_checked_folding_refuses_a_fold_that_changes_the_output(monkeypatch):
    def broken(model, remove_bn=False):
        with torch.no_grad():
            model.a.conv.weight.mul_(2.0)
        return 1, 0, 0

    monkeypatch.setattr(bn_fold, "fold_batchnorm", broken)

    with pytest.raises(RuntimeError, match="changed the float model's output"):
        bn_fold.fold_batchnorm_checked(_net(), (1, 3, 16, 16))


def test_bn_without_running_stats_is_refused():
    conv = nn.Conv2d(3, 4, 1)
    bn = nn.BatchNorm2d(4, track_running_stats=False)

    with pytest.raises(ValueError, match="running statistics"):
        bn_fold.fold_conv_bn(conv, bn)


class OtherNorm(nn.BatchNorm2d):
    """A BN subclass whose post-normalization behaviour is not known."""


class OtherContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1, bias=False)
        self.bn = _randomize_bn(OtherNorm(4))

    def forward(self, x):
        return self.bn(self.conv(x))


OtherContainer.__name__ = "ConvNormAct"


def test_a_bn_of_unknown_type_is_folded_but_kept_as_identity_not_guessed():
    torch.manual_seed(0)
    net = OtherContainer().eval()
    x = torch.randn(1, 3, 5, 5)
    with torch.no_grad():
        before = net(x)
        assert bn_fold.fold_batchnorm(net, remove_bn=True) == (1, 0, 1)
        after = net(x)

    assert isinstance(net.bn, OtherNorm)  # still there, now an identity
    assert torch.allclose(before, after, rtol=1e-5, atol=1e-5)


def test_the_kept_activation_no_longer_runs_in_place_on_the_conv_output():
    net = _net()
    assert net.a.bn.act.inplace is True
    bn_fold.fold_batchnorm(net, remove_bn=True)

    act = net.a.bn[-1]
    assert act.inplace is False
    conv_out = net.a.conv(torch.randn(1, 3, 8, 8))
    snapshot = conv_out.clone()
    with torch.no_grad():
        net.a.bn(conv_out)
    assert torch.equal(conv_out, snapshot)  # the conv's own output tensor is left untouched
