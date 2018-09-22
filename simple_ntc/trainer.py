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

    def load_model(self, fn):
        self.best = torch.load(fn)

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
        total_loss, total_param_norm, total_grad_norm = 0, 0, 0
        avg_loss, avg_param_norm, avg_grad_norm = 0, 0, 0

        progress_bar = tqdm(train, 
                            desc='Training: ', 
                            unit='batch'
                            ) if verbose is VERBOSE_BATCH_WISE else enumerate(train)
        for idx, mini_batch in enumerate(progress_bar):
            x, y = mini_batch.text, mini_batch.label
            optimizer.zero_grad()

            y_hat = self.model(x)

            loss = self.get_loss(y_hat, y)
            loss.backward()

            total_loss += loss
            total_param_norm += utils.get_parameter_norm(self.model.parameters())
            total_grad_norm += utils.get_grad_norm(self.model.parameters())

            avg_loss = total_loss / (idx + 1)
            avg_param_norm = total_param_norm / (idx + 1)
            avg_grad_norm = total_grad_norm / (idx + 1)

            if verbose is VERBOSE_BATCH_WISE:
                progress_bar.set_postfix_str('|param|=%.2f |g_param|=%.2f loss=%.4e' % (avg_param_norm,
                                                                                        avg_grad_norm,
                                                                                        avg_loss
                                                                                        ))

            optimizer.step()

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
        optimizer = torch.optim.Adam(self.model.parameters())

        lowest_loss = float('Inf')
        loewst_after = 0

        progress_bar = tqdm(range(n_epochs), 
                            desc='Training: ', 
                            unit='epoch'
                            ) if verbose is VERBOSE_EPOCH_WISE else range(n_epochs)
        for idx in progress_bar:
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
            _, avg_valid_loss = self.predict(valid, 
                                             verbose=verbose
                                             )

            if verbose is VERBOSE_EPOCH_WISE:
                progress_bar.set_postfix_str('|param|=%.2f |g_param|=%.2f train_loss=%.4e valid_loss=%.4e min_valid_loss=%.4e' % (float(avg_param_norm),
                                                                                                                                      float(avg_grad_norm),
                                                                                                                                      float(avg_train_loss),
                                                                                                                                      float(avg_valid_loss),
                                                                                                                                      float(lowest_loss)
                                                                                                                                      ))

            if avg_valid_loss < lowest_loss:
                lowest_loss = avg_valid_loss
                loewst_after = 0

                self.best = {'model': self.model.state_dict(),
                             'optim': optimizer,
                             'epoch': idx,
                             'lowest_loss': lowest_loss
                             }
            else:
                loewst_after += 1

                if loewst_after >= early_stop and early_stop > 0:
                    break
        if verbose is VERBOSE_EPOCH_WISE:
            progress_bar.close()

    def predict(self, 
                valid, 
                crit=None, 
                batch_size=256, 
                return_numpy=True, 
                verbose=VERBOSE_SILENT
                ):
        with torch.no_grad():
            total_loss = 0
            progress_bar = tqdm(valid, 
                                desc='Validation: ', 
                                unit='batch'
                                ) if verbose is VERBOSE_BATCH_WISE else enumerate(valid)

            y_hats = []
            self.model.eval()
            for idx, mini_batch in enumerate(progress_bar):
                x, y = mini_batch.text, mini_batch.label
                y_hat = self.model(x)
                
                loss = self.get_loss(y_hat, y, crit)

                total_loss += loss
                avg_loss = total_loss / (idx + 1)
                y_hats += [y_hat]

                if verbose is VERBOSE_BATCH_WISE:
                    progress_bar.set_postfix_str('valid_loss=%.4e' % avg_loss)
            self.model.train()

            if verbose is VERBOSE_BATCH_WISE:
                progress_bar.close()

            y_hats = torch.cat(y_hats, dim=0)
            if return_numpy:
                y_hats = y_hats.numpy()

            return y_hats, avg_loss
