import numpy as np
import torch

from torch import optim
from ignite.engine import Engine, Events


class TextClassification():

    def __init__(self, config):
        self.config = config

    @staticmethod
    def step(engine, mini_batch):
        from utils import get_grad_norm, get_parameter_norm

        engine.model.train()
        engine.optimizer.zero_grad()

        x, y = mini_batch.text, mini_batch.label

        y_hat = engine.model(x)
        loss = engine.crit(y_hat, y)
        loss.backward()

        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        engine.optimizer.step()

        return float(loss), p_norm, g_norm

    @staticmethod
    def validate(engine, mini_batch):
        from utils import get_grad_norm, get_parameter_norm

        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch.text, mini_batch.label

            y_hat = engine.model(x)
            loss = engine.crit(y_hat, y)

        return float(loss)

    @staticmethod
    def attach(trainer, evaluator, verbose=2):
        from ignite.engine import Events
        from ignite.metrics import RunningAverage
        from ignite.contrib.handlers.tqdm_logger import ProgressBar

        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
        RunningAverage(output_transform=lambda x: x[1]).attach(trainer, '|param|')
        RunningAverage(output_transform=lambda x: x[2]).attach(trainer, '|g_param|')

        if verbose >= 2:
            pbar = ProgressBar()
            pbar.attach(trainer, ['|param|', '|g_param|', 'loss'])

        if verbose >= 1:
            @trainer.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics['|param|']
                avg_g_norm = engine.state.metrics['|g_param|']
                avg_loss = engine.state.metrics['loss']

                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e}'.format(engine.state.epoch, avg_p_norm, avg_g_norm, avg_loss))

        RunningAverage(output_transform=lambda x: x).attach(evaluator, 'loss')

        if verbose >= 2:
            pbar = ProgressBar()
            pbar.attach(evaluator, ['loss'])

        if verbose >= 1:
            @evaluator.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_loss = engine.state.metrics['loss']
                print('Validation - loss={:.4e} lowest_loss={:.4e}'.format(avg_loss, engine.lowest_loss))

    def train(self, model, crit, train_loader, valid_loader):
        optimizer = optim.Adam(model.parameters())

        trainer = Engine(TextClassification.step)
        trainer.model, trainer.crit, trainer.optimizer = model, crit, optimizer

        evaluator = Engine(TextClassification.validate)
        evaluator.model, evaluator.crit = model, crit
        evaluator.lowest_loss = np.inf

        TextClassification.attach(trainer, evaluator, verbose=self.config.verbose)

        def run_validation(engine, evaluator, valid_loader):
            evaluator.run(valid_loader, max_epochs=1)

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, run_validation, evaluator, valid_loader
        )

        @evaluator.on(Events.EPOCH_COMPLETED)
        def check_loss(engine):
            from copy import deepcopy

            loss = float(engine.state.metrics['loss'])
            if loss <= engine.lowest_loss:
                engine.lowest_loss = loss
                engine.best_model = deepcopy(engine.model.state_dict())

        trainer.run(train_loader, max_epochs=self.config.n_epochs)
        model.load_state_dict(evaluator.best_model)

        return model
