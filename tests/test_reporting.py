import pytest
from phase1.reporting import validate_comparison


def test_comparison_rejects_different_decoding_settings():
    a = dict(quantizer='rtn', bits=3, generation={'temperature': 1}, input_sha256={'queries': 'a'})
    b = dict(quantizer='awq', bits=3, generation={'temperature': .5}, input_sha256={'queries': 'a'})
    with pytest.raises(ValueError, match='generation'):
        validate_comparison([a, b])


def test_comparison_rejects_different_query_content_even_same_path():
    a = dict(quantizer='rtn', bits=3, input_sha256={'queries': 'a'})
    b = dict(quantizer='rtn', bits=4, input_sha256={'queries': 'b'})
    with pytest.raises(ValueError, match='input_sha256'):
        validate_comparison([a, b])
