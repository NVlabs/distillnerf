from mmcv.runner import EpochBasedRunner
import time, random
import torch
# from .builder import RUNNERS

from mmcv.runner.builder import RUNNERS


@RUNNERS.register_module()
class EpochBasedRunnerAnomoly(EpochBasedRunner):
    '''
        Debugging use: EpochBasedRunner with torch.autograd.detect_anomaly(True) in train method.
    '''
    
    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            with torch.autograd.detect_anomaly(True):
                self.data_batch = data_batch
                self._inner_iter = i
                self.call_hook('before_train_iter')
                self.run_iter(data_batch, train_mode=True, **kwargs)
                self.call_hook('after_train_iter')
                del self.data_batch
                self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

@RUNNERS.register_module()
class EpochBasedRunnerValFirst(EpochBasedRunner):
    '''
        start validation before training
    '''
    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        # import pdb; pdb.set_trace()
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1

            
            # due to 4h runtime limit on server, the model can barely be evaluated
            # thus for every 2 epoch after 8 epochs, we'll do evaluation before training
            if self._epoch > 8 and self._epoch % 2 == 0: break
            # break

        self.call_hook('after_train_epoch')
        self._epoch += 1

@RUNNERS.register_module()
class EpochBasedRunnerValFirstPassIter(EpochBasedRunner):
    '''
        start validation before training, pass current iteration to model wrapper
    '''

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_train_iter')
            kwargs['iteration'] = self._iter
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1
            
            # due to 4h runtime limit on server, the model can barely be evaluated
            # thus for every 2 epoch after 8 epochs, we'll do evaluation before training
            if self._epoch > 8 and self._epoch % 2 == 0: break

        self.call_hook('after_train_epoch')
        self._epoch += 1