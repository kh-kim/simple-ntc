from tqdm import tqdm
import torch

import utils

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class Trainer():

    def __init__(self, model, crit):
        self.model = model
        self.crit = crit

        super().__init__()

        self.best = {}

    def get_best_model(self):
        self.model.load_state_dict(self.best['model'])

        return self.model

    def get_loss(self, y_hat, y, crit=None):
        crit = self.crit if crit is None else crit
        loss = crit(y_hat, y)

        return loss

    def train_epoch(self, 
                    train, 
                    optimizer, 
                    batch_size=64, 
                    verbose=VERBOSE_SILENT
                    ):
        '''
        Train an epoch with given train iterator and optimizer.
        '''
        total_loss, total_param_norm, total_grad_norm = 0, 0, 0
        avg_loss, avg_param_norm, avg_grad_norm = 0, 0, 0
        sample_cnt = 0

        progress_bar = tqdm(train, 
                            desc='Training: ', 
                            unit='batch'
                            ) if verbose is VERBOSE_BATCH_WISE else train
        # Iterate whole train-set.
        for idx, mini_batch in enumerate(progress_bar):
            x, y = mini_batch.text, mini_batch.label
            # Don't forget make grad zero before another back-prop.
            optimizer.zero_grad()

            y_hat = self.model(x)

            loss = self.get_loss(y_hat, y)
            loss.backward()

            total_loss += loss
            total_param_norm += utils.get_parameter_norm(self.model.parameters())
            total_grad_norm += utils.get_grad_norm(self.model.parameters())

            # Caluclation to show status
            avg_loss = total_loss / (idx + 1)
            avg_param_norm = total_param_norm / (idx + 1)
            avg_grad_norm = total_grad_norm / (idx + 1)

            if verbose is VERBOSE_BATCH_WISE:
                progress_bar.set_postfix_str('|param|=%.2f |g_param|=%.2f loss=%.4e' % (avg_param_norm,
                                                                                        avg_grad_norm,
                                                                                        avg_loss
                                                                                        ))

            optimizer.step()

            sample_cnt += mini_batch.text.size(0)
            if sample_cnt >= len(train.dataset.examples):
                break

        if verbose is VERBOSE_BATCH_WISE:
            progress_bar.close()

        return avg_loss, avg_param_norm, avg_grad_norm

    def train(self, 
              train, 
              valid, 
              batch_size=64,
              n_epochs=100, 
              early_stop=-1, 
              verbose=VERBOSE_SILENT
              ):
        '''
        Train with given train and valid iterator until n_epochs.
        If early_stop is set, 
        early stopping will be executed if the requirement is satisfied.
        '''
        optimizer = torch.optim.Adam(self.model.parameters())

        lowest_loss = float('Inf')
        lowest_after = 0

        progress_bar = tqdm(range(n_epochs), 
                            desc='Training: ', 
                            unit='epoch'
                            ) if verbose is VERBOSE_EPOCH_WISE else range(n_epochs)
        for idx in progress_bar:  # Iterate from 1 to n_epochs
            if verbose > VERBOSE_EPOCH_WISE:
                print('epoch: %d/%d\tmin_valid_loss=%.4e' % (idx + 1, 
                                                             len(progress_bar), 
                                                             lowest_loss
                                                             ))
            avg_train_loss, avg_param_norm, avg_grad_norm = self.train_epoch(train, 
                                                                             optimizer, 
                                                                             batch_size=batch_size, 
                                                                             verbose=verbose
                                                                             )
            _, avg_valid_loss = self.validate(valid, 
                                              verbose=verbose
                                              )

            # Print train status with different verbosity.
            if verbose is VERBOSE_EPOCH_WISE:
                progress_bar.set_postfix_str('|param|=%.2f |g_param|=%.2f train_loss=%.4e valid_loss=%.4e min_valid_loss=%.4e' % (float(avg_param_norm),
                                                                                                                                  float(avg_grad_norm),
                                                                                                                                  float(avg_train_loss),
                                                                                                                                  float(avg_valid_loss),
                                                                                                                                  float(lowest_loss)
                                                                                                                                  ))

            if avg_valid_loss < lowest_loss:
                # Update if there is an improvement.
                lowest_loss = avg_valid_loss
                lowest_after = 0

                self.best = {'model': self.model.state_dict(),
                             'optim': optimizer,
                             'epoch': idx,
                             'lowest_loss': lowest_loss
                             }
            else:
                lowest_after += 1

                if lowest_after >= early_stop and early_stop > 0:
                    break
        if verbose is VERBOSE_EPOCH_WISE:
            progress_bar.close()

    def validate(self, 
                 valid, 
                 crit=None, 
                 batch_size=256, 
                 verbose=VERBOSE_SILENT
                 ):
        '''
        Validate a model with given valid iterator.
        '''
        # We don't need to back-prop for these operations.
        with torch.no_grad():
            total_loss, total_correct, sample_cnt = 0, 0, 0
            progress_bar = tqdm(valid, 
                                desc='Validation: ', 
                                unit='batch'
                                ) if verbose is VERBOSE_BATCH_WISE else valid

            y_hats = []
            self.model.eval()
            # Iterate for whole valid-set.
            for idx, mini_batch in enumerate(progress_bar):
                x, y = mini_batch.text, mini_batch.label
                y_hat = self.model(x)
                # |y_hat| = (batch_size, n_classes)
                
                loss = self.get_loss(y_hat, y, crit)

                total_loss += loss
                sample_cnt += mini_batch.text.size(0)
                total_correct += float(y_hat.topk(1)[1].view(-1).eq(y).sum())

                avg_loss = total_loss / (idx + 1)
                y_hats += [y_hat]

                if verbose is VERBOSE_BATCH_WISE:
                    progress_bar.set_postfix_str('valid_loss=%.4e accuarcy=%.4f' % (avg_loss, total_correct / sample_cnt))

                if sample_cnt >= len(valid.dataset.examples):
                    break
            self.model.train()

            if verbose is VERBOSE_BATCH_WISE:
                progress_bar.close()

            y_hats = torch.cat(y_hats, dim=0)

            return y_hats, avg_loss
