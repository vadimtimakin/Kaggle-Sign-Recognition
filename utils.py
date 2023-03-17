import os
import torch
import random
import sklearn
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from objects.optimizer import *
from objects.custom_functions import *
from objects.scheduler import GradualWarmupSchedulerV2


def get_optimizer(config, model):
    '''Get PyTorch optimizer'''

    if config.optimizer.name.startswith('/custom/'):
        optimizer = globals()[config.optimizer.name[8:]](model.parameters(), **config.optimizer.params)
    else:
        optimizer = getattr(torch.optim, config.optimizer.name)(model.parameters(), **config.optimizer.params)

    return optimizer


def get_scheduler(config, optimizer):
    '''Get PyTorch scheduler'''

    if config.scheduler.name.startswith('/custom/'):
        scheduler = globals()[config.scheduler.name[8:]](optimizer, **config.scheduler.params)
    else:
        scheduler = getattr(torch.optim.lr_scheduler, config.scheduler.name)(optimizer, **config.scheduler.params)
    
    if config.training.warmup_scheduler:
        final_scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=config.training.warmup_multiplier,
                                                 total_epoch=config.training.warmup_epochs, after_scheduler=scheduler)
        return final_scheduler
    else:
        return scheduler

    
def get_loss(config):
    '''Get PyTorch loss function.'''

    if config.loss.name.startswith('/custom/'):
        loss = globals()[config.loss.name[8:]](**config.loss.params)
    else:
        loss = getattr(nn, config.loss.name)(**config.loss.params)

    return loss


def get_metric(config, y_true, y_pred):
    '''Calculate metric.'''
    
    predictions = np.argmax(y_pred, axis=1)

    if config.metric.name.startswith('/custom/'):
        score = globals()[config.metric.name[8:]](y_true, predictions, **config.metric.params)
    else:
        score = getattr(sklearn.metrics, config.metric.name)(y_true, predictions, **config.metric.params)
    
    return score


def set_seed(seed: int):
    '''Set a random seed for complete reproducibility.'''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_model(config, model, epoch, current_metric, optimizer, epochs_since_improvement, 
                       name, scheduler, scaler, best_metric):
    '''Save PyTorch model.'''

    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'metric': current_metric,
        'optimizer': optimizer.state_dict(),
        'epochs_since_improvement': epochs_since_improvement,
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'best_metric': best_metric,
    }, os.path.join(config.paths.path_to_checkpoints, name))


def draw_plots(train_losses, val_losses, metrics, lr_changes):
    '''Draw plots of losses, metrics and learning rate changes.'''

    # Learning rate changes
    plt.plot(range(len(lr_changes)), lr_changes, label='Learning Rate')
    plt.legend()
    plt.title('Learning rate changes')
    plt.show()

    # Validation and train losses
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Changes of validation and train losses')
    plt.show()

    # Metric changes
    plt.plot(range(len(metrics)), metrics, label='Metric')
    plt.legend()
    plt.title('Metric changes')
    plt.show()

def print_report(t, train_loss, val_loss, metric, best_metric):
    '''Print report of one epoch.'''

    print(f'Time: {t} s')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Val Loss: {val_loss:.4f}')
    print(f'Current Metric: {metric:.4f}')
    print(f'Best Metric: {best_metric:.4f}')


def save_log(path, epoch, train_loss, val_loss, best_metric):
    '''Save log of one epoch.'''

    with open(path, 'a') as file:
        file.write('epoch: ' + str(epoch) + ' train_loss: ' + str(round(train_loss, 5)) + 
                   ' val_loss: ' + str(round(val_loss, 5)) + ' best_metric: ' + 
                   str(round(best_metric, 5)) + '\n')