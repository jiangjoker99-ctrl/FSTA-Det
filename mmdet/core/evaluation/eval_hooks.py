# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import math
import os
import os.path as osp

import mmcv
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


def _is_valid_key_score(key_score):
    """Accept finite scores, including the legitimate value zero."""
    if key_score is None:
        return False
    try:
        return math.isfinite(float(key_score))
    except (TypeError, ValueError):
        return False


def _save_best_checkpoint_safely(hook, runner, key_score):
    """Save the new best before removing the previous checkpoint.

    FSTA-Det uses a local work directory.  Saving through a temporary filename
    followed by ``os.replace`` prevents a failed write from destroying the
    previous best model.
    """
    if hook.by_epoch:
        current = f'epoch_{runner.epoch + 1}'
        current_type, current_time = 'epoch', runner.epoch + 1
    else:
        current = f'iter_{runner.iter + 1}'
        current_type, current_time = 'iter', runner.iter + 1

    messages = runner.meta.setdefault('hook_msgs', {})
    best_score = messages.get('best_score', hook.init_value_map[hook.rule])
    if not hook.compare_func(key_score, best_score):
        return

    previous_path = hook.best_ckpt_path
    previous_score = messages.get('best_score', None)
    checkpoint_name = f'best_{hook.key_indicator}_{current}.pth'
    final_path = hook.file_client.join_path(hook.out_dir, checkpoint_name)
    temporary_name = f'.{checkpoint_name}.tmp'
    temporary_path = hook.file_client.join_path(hook.out_dir, temporary_name)

    messages['best_score'] = key_score
    messages['best_ckpt'] = final_path
    try:
        runner.save_checkpoint(
            hook.out_dir,
            filename_tmpl=temporary_name,
            create_symlink=False)
        os.replace(temporary_path, final_path)
    except Exception:
        if osp.isfile(temporary_path):
            os.remove(temporary_path)
        if previous_score is None:
            messages.pop('best_score', None)
        else:
            messages['best_score'] = previous_score
        if previous_path is None:
            messages.pop('best_ckpt', None)
        else:
            messages['best_ckpt'] = previous_path
        raise

    hook.best_ckpt_path = final_path
    if (previous_path and previous_path != final_path
            and hook.file_client.isfile(previous_path)):
        hook.file_client.remove(previous_path)
        runner.logger.info(
            f'The previous best checkpoint {previous_path} was removed')
    runner.logger.info(f'Now best checkpoint is saved as {checkpoint_name}.')
    runner.logger.info(
        f'Best {hook.key_indicator} is {key_score:0.4f} at '
        f'{current_time} {current_type}.')


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class EvalHook(BaseEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        # BaseEvalHook replaces a missing test_fn with MMCV's generic test
        # function.  That function does not pass ``rescale=True`` to an MMDet
        # detector, so its boxes remain in resized-image coordinates.  Keep
        # track of whether the caller supplied a custom function and use
        # MMDetection's test function for the default path.
        self._use_mmdet_single_gpu_test = kwargs.get('test_fn') is None
        super(EvalHook, self).__init__(*args, **kwargs)

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        if self._use_mmdet_single_gpu_test:
            from mmdet.apis import single_gpu_test
            results = single_gpu_test(
                runner.model, self.dataloader, show=False)
        else:
            results = self.test_fn(runner.model, self.dataloader)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and _is_valid_key_score(key_score):
            _save_best_checkpoint_safely(self, runner, key_score)


# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.
class DistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(DistEvalHook, self).__init__(*args, **kwargs)

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmdet.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            # the key_score may be `None` so it needs to skip
            # the action to save the best checkpoint
            if self.save_best and _is_valid_key_score(key_score):
                _save_best_checkpoint_safely(self, runner, key_score)
