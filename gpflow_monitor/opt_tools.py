# Copyright 2017 Mark van der Wilk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import enum
import glob
import os

import gpflow
import numpy as np
import tensorflow as tf

from . import timer


@enum.unique
class Trigger(enum.Enum):
    TOTAL_TIME = 1
    OPTIMISATION_TIME = 2
    ITER = 3


class Task(metaclass=abc.ABCMeta):
    def __init__(self, sequence, trigger: Trigger):
        """
        :param sequence: Sequence of times at which Task needs to be executed. If None, the task
        will only be executed at the end of optimisation.
        :param trigger: Timer in `ManagedOptimisation` to use for determining execution.
        """
        # Construct the object before compile. You can do graph manipulations here.
        self._seq = sequence
        self._trigger = trigger
        self._next = next(self._seq) if self._seq is not None else np.inf
        self.verbose = False
        self.time_spent = timer.Stopwatch()
        self.times_called = timer.ElapsedTracker()

    @abc.abstractmethod
    def _event_handler(self, manager):
        raise NotImplementedError

    def __call__(self, manager, force_run):
        self.time_spent.start()
        if manager.timers[self._trigger].elapsed >= self._next or force_run:
            # if self.verbose:
            #     print("%.2f - %s" % (manager.timers[self._trigger].elapsed, str(self)))
            self._event_handler(manager)
            self.times_called.add(1)

        # Move to the next trigger time, and make sure it's after this current iteration
        while self._next <= manager.timers[self._trigger].elapsed:
            self._next = next(self._seq)
        self.time_spent.stop()


class StoreSession(Task):
    def __init__(self, sequence, trigger: Trigger, session: tf.Session, hist_path, saver=None,
                 restore_path=None):
        super().__init__(sequence, trigger)
        self.hist_path = hist_path
        self.restore_path = restore_path
        self.saver = tf.train.Saver(max_to_keep=3) if saver is None else saver
        self.session = session

        restore_path = self.restore_path
        if restore_path is None:
            if len(glob.glob(self.hist_path + "-*")) > 0:
                # History exists
                latest_step = np.max([int(os.path.splitext(os.path.basename(x))[0].split('-')[-1])
                                      for x in glob.glob(self.hist_path + "-*")])
                restore_path = self.hist_path + "-%i" % latest_step

        if restore_path is not None:
            print("Restoring session from `%s`." % restore_path)
            self.saver.restore(session, restore_path)

    def _event_handler(self, manager):
        self.saver.save(self.session, self.hist_path, global_step=manager.global_step)


class TensorBoard(Task):
    def __init__(self, sequence, trigger: Trigger, tensors: list,
                 file_writer: tf.summary.FileWriter):
        super().__init__(sequence, trigger)
        self.summary = tf.summary.merge(tensors)
        self.file_writer = file_writer

    def _event_handler(self, manager):
        summary, step = manager.session.run([self.summary, manager.global_step])
        self.file_writer.add_summary(summary, step)


class ModelTensorBoard(TensorBoard):
    def __init__(self, sequence, trigger: Trigger, model: gpflow.models.Model,
                 file_writer: tf.summary.FileWriter, parameters=None, additional_summaries=None):
        """
        Creates a Task that creates a sensible TensorBoard for a model.
        :param sequence:
        :param trigger:
        :param model:
        :param file_writer:
        :param parameters: List of `gpflow.Parameter` objects to send to TensorBoard if they are
        scalar. If None, all scalars will be sent to TensorBoard.
        :param additional_summaries: List of Summary objects to send to TensorBoard.
        """
        self.model = model
        all_summaries = [] if additional_summaries is None else additional_summaries
        if parameters is None:
            all_summaries += [tf.summary.scalar(p.full_name, p.constrained_tensor)
                              for p in model.parameters if len(p.shape) == 0]
            all_summaries.append(tf.summary.scalar("likelihood", model._likelihood_tensor))
        else:
            all_summaries += [tf.summary.scalar(p.full_name, p.constrained_tensor)
                              for p in parameters if p.size == 1]

        super().__init__(sequence, trigger, all_summaries, file_writer)


class LmlTensorBoard(ModelTensorBoard):
    def __init__(self, sequence, trigger: Trigger, model, file_writer, minibatch_size=100,
                 verbose=True):
        super().__init__(sequence, trigger, model, file_writer)
        self.minibatch_size = minibatch_size
        self._full_lml = tf.placeholder(gpflow.settings.tf_float, shape=())
        self.summary = tf.summary.scalar("full_lml", self._full_lml)
        self.verbose = verbose

    def _event_handler(self, manager):
        m = manager.model
        with gpflow.decors.params_as_tensors_for(m):
            tfX, tfY = m.X, m.Y

        if self.verbose:
            import tqdm
            wrapper = tqdm.tqdm
        else:
            wrapper = lambda x: x

        lml = 0.0
        num_batches = -(-len(m.X._value) // self.minibatch_size)  # round up
        for mb in wrapper(range(num_batches)):
            start = mb * self.minibatch_size
            finish = (mb + 1) * self.minibatch_size
            Xmb = m.X._value[start:finish, :]
            Ymb = m.Y._value[start:finish, :]
            mb_lml = m.compute_log_likelihood(feed_dict={tfX: Xmb, tfY: Ymb})
            lml += mb_lml * len(Xmb)
        lml = lml / len(m.X._value)

        summary, step = manager.session.run([self.summary, manager.global_step],
                                             feed_dict={self._full_lml: lml})
        print("Full lml: %f (%.2e)" % (lml, lml))
        self.file_writer.add_summary(summary, step)


class PrintTimings(Task):
    def _event_handler(self, manager):
        current_iter = manager.timers[Trigger.ITER].elapsed
        if current_iter == 0:
            opt_iter = 0.0
            total_iter = 0.0
            last_iter = 0.0
        else:
            opt_iter = current_iter / manager.timers[Trigger.OPTIMISATION_TIME].elapsed
            total_iter = current_iter / manager.timers[Trigger.TOTAL_TIME].elapsed
            last_iter = (0.0 if not hasattr(self, '_last_iter')
                         else (current_iter - self._last_iter) / self._last_iter_timer.elapsed)

        global_step_eval = manager.session.run(manager.global_step)
        print("\r%i, %i:\t%.2f optimisation iter/s\t%.2f total iter/s\t%.2f last iter/s" %
              (current_iter, global_step_eval, opt_iter, total_iter, last_iter), end='')

        self._last_iter = current_iter
        self._last_iter_timer = timer.Stopwatch().start()


class FunctionCallback(Task):
    def __init__(self, sequence, trigger, func):
        super().__init__(sequence, trigger)
        self._func = func

    def _event_handler(self, manager):
        self._func(manager.model)


class PrintAllTimings(PrintTimings):
    def _event_handler(self, manager):
        super()._event_handler(manager)
        manager.print_timings()


class ManagedOptimisation:
    def __init__(self, model: gpflow.models.Model, optimiser: gpflow.training.optimizer.Optimizer,
                 global_step, session=None, var_list=None):
        self._opt_method = optimiser

        self.session = model.enquire_session(session)

        # Setup timers
        total_time = timer.Stopwatch()
        optimisation_time = timer.Stopwatch()
        iter_count = timer.ElapsedTracker()
        self.timers = {Trigger.TOTAL_TIME: total_time,
                       Trigger.OPTIMISATION_TIME: optimisation_time,
                       Trigger.ITER: iter_count}
        self.tasks = []
        self.model = model
        self.global_step = global_step


        # Setup optimiser variables etc
        self._opt_method.minimize(model, session=self.session,
            maxiter=0, global_step=global_step, var_list=var_list)

    def callback(self, force_run):
        with self.timers[Trigger.OPTIMISATION_TIME].pause():
            for task in self.tasks:
                task(self, force_run)

    def minimize(self, maxiter=0):
        try:
            [t.start() for t in self.timers.values()]
            while self.timers[Trigger.ITER].elapsed < maxiter:
                self.session.run([self._opt_method.minimize_operation])  # GPflow internal
                self.timers[Trigger.ITER].add(1)
                self.callback(force_run=False)
        finally:
            self.model.anchor(self.session)
            self.callback(force_run=True)
            [t.stop() for t in self.timers.values()]

    def print_timings(self):
        print("")
        print("Number of iterations  : %i" % self.timers[Trigger.ITER].elapsed)
        print("Total time spent      : %.2fs" % self.timers[Trigger.TOTAL_TIME].elapsed)
        print("Time spent optimising : %.2fs" % self.timers[Trigger.OPTIMISATION_TIME].elapsed)
        for t in self.tasks:
            print("Time spent with %s: %.2fs (%i)" % (str(t), t.time_spent.elapsed,
                                                      t.times_called.elapsed))


def seq_exp_lin(growth, max, start=1.0, start_jump=None):
    """
    Returns an iterator that constructs a sequence beginning with `start`, growing exponentially:
    the step size starts out as `start_jump` (if given, otherwise `start`), multiplied by `growth`
    in each step. Once `max` is reached, growth will be linear with `max` step size.
    """
    start_jump = start if start_jump is None else start_jump
    gap = start_jump
    last = start - start_jump
    while 1:
        yield gap + last
        last = last + gap
        gap = min(gap * growth, max)
