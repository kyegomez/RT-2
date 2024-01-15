import pytest
import torch
from rt2.model import RT2


@pytest.fixture
def rt2():
    return RT2()


@pytest.fixture
def img():
    return torch.rand((1, 3, 256, 256))


@pytest.fixture
def text():
    return torch.randint(0, 20000, (1, 1024))


def test_init(rt2):
    assert isinstance(rt2, RT2)


def test_forward(rt2, img, text):
    output = rt2(img, text)
    assert output.shape == (1, 1024, 20000)


def test_forward_different_img_shape(rt2, text):
    img = torch.rand((2, 3, 256, 256))
    output = rt2(img, text)
    assert output.shape == (2, 1024, 20000)


def test_forward_different_text_length(rt2, img):
    text = torch.randint(0, 20000, (1, 512))
    output = rt2(img, text)
    assert output.shape == (1, 512, 20000)


def test_forward_different_num_tokens(rt2, img, text):
    rt2.decoder.num_tokens = 10000
    output = rt2(img, text)
    assert output.shape == (1, 1024, 10000)


def test_forward_different_max_seq_len(rt2, img, text):
    rt2.decoder.max_seq_len = 512
    output = rt2(img, text)
    assert output.shape == (1, 512, 20000)


def test_forward_exception(rt2, img):
    with pytest.raises(Exception):
        rt2(img)


def test_forward_no_return_embeddings(rt2, img, text):
    rt2.encoder.return_embeddings = False
    with pytest.raises(Exception):
        rt2(img, text)


def test_forward_different_encoder_dim(rt2, img, text):
    rt2.encoder.dim = 256
    output = rt2(img, text)
    assert output.shape == (1, 1024, 20000)


def test_forward_different_encoder_depth(rt2, img, text):
    rt2.encoder.depth = 3
    output = rt2(img, text)
    assert output.shape == (1, 1024, 20000)


def test_forward_different_encoder_heads(rt2, img, text):
    rt2.encoder.heads = 4
    output = rt2(img, text)
    assert output.shape == (1, 1024, 20000)


def test_forward_different_decoder_dim(rt2, img, text):
    rt2.decoder.dim = 256
    output = rt2(img, text)
    assert output.shape == (1, 1024, 20000)


def test_forward_different_decoder_depth(rt2, img, text):
    rt2.decoder.depth = 3
    output = rt2(img, text)
    assert output.shape == (1, 1024, 20000)


def test_forward_different_decoder_heads(rt2, img, text):
    rt2.decoder.heads = 4
    output = rt2(img, text)
    assert output.shape == (1, 1024, 20000)


def test_forward_different_alibi_num_heads(rt2, img, text):
    rt2.decoder.alibi_num_heads = 2
    output = rt2(img, text)
    assert output.shape == (1, 1024, 20000)
