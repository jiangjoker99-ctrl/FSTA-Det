from types import SimpleNamespace
from unittest.mock import Mock, patch

from torch.utils.data import DataLoader

from mmdet.core.evaluation.eval_hooks import EvalHook


def _runner():
    return SimpleNamespace(
        model=object(), log_buffer=SimpleNamespace(output={}))


def _make_runnable(hook):
    hook._should_evaluate = Mock(return_value=True)
    hook.evaluate = Mock(return_value=None)
    hook.save_best = None


def test_default_eval_uses_mmdet_single_gpu_test():
    loader = DataLoader([0])
    hook = EvalHook(loader, interval=1)
    _make_runnable(hook)
    runner = _runner()

    with patch('mmdet.apis.single_gpu_test', return_value=['result']) as test_fn:
        hook._do_evaluate(runner)

    test_fn.assert_called_once_with(runner.model, loader, show=False)
    hook.evaluate.assert_called_once_with(runner, ['result'])


def test_custom_eval_function_is_preserved():
    loader = DataLoader([0])
    custom_test = Mock(return_value=['custom-result'])
    hook = EvalHook(loader, interval=1, test_fn=custom_test)
    _make_runnable(hook)
    runner = _runner()

    hook._do_evaluate(runner)

    custom_test.assert_called_once_with(runner.model, loader)
    hook.evaluate.assert_called_once_with(runner, ['custom-result'])
