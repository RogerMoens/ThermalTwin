from pathlib import Path

import numpy as np
import pytest

from models.base_predictor import BasePredictor, FlattenMixin


class _Dummy(BasePredictor):
    """Minimal concrete BasePredictor used to exercise the shared helpers."""

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        pass

    def predict(self, X):
        pass

    def save(self, folder):
        pass

    @classmethod
    def load(cls, folder):
        pass


def test_base_predictor_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BasePredictor()


def test_incomplete_subclass_cannot_be_instantiated():
    class Incomplete(BasePredictor):
        def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
            pass
        # predict/save/load intentionally left unimplemented

    with pytest.raises(TypeError):
        Incomplete()


def test_prepare_folder_creates_nested_directories(tmp_path):
    target = tmp_path / "a" / "b" / "c"

    result = _Dummy._prepare_folder(target)

    assert result == target
    assert target.is_dir()


def test_prepare_folder_accepts_string_path(tmp_path):
    target = tmp_path / "x"

    result = _Dummy._prepare_folder(str(target))

    assert isinstance(result, Path)
    assert result.is_dir()


def test_prepare_folder_is_idempotent_for_existing_directory(tmp_path):
    target = tmp_path / "already_here"
    target.mkdir()

    result = _Dummy._prepare_folder(target)

    assert result.is_dir()


def test_flatten_mixin_reshapes_to_2d():
    class Flattener(FlattenMixin):
        pass

    X = np.zeros((5, 3, 4))
    flat = Flattener()._flatten(X)

    assert flat.shape == (5, 12)
