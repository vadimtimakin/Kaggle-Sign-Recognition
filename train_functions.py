import os
import gc
import time 
import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.cuda import amp

from data import get_loaders
from objects.model import BasedPartyNet
from utils import save_log, save_model, print_report, draw_plots
from utils import get_loss, get_metric,  get_optimizer, get_scheduler


def train(config, model, train_loader, optimizer, scheduler, loss_function, epoch, scaler):
    '''Train loop.'''

    if config.logging.prints:
        print('Training')

    model.train()

    if config.model.freeze_batchnorms:
        for name, child in (model.named_children()):
            if name.find('BatchNorm') != -1:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
    
    total_loss = 0.0

    if config.logging.prints:
        train_loader = tqdm(train_loader)

    for step, batch in enumerate(train_loader):
        batch["features"] = [i.to(config.training.device) for i in batch["features"]]
        batch["labels"] = batch["labels"].to(config.training.device)

        if not config.training.gradient_accumulation:
            optimizer.zero_grad()
        
        if config.training.mixed_precision:
            with amp.autocast():
                outputs = model(batch["features"])
                loss = loss_function(outputs, batch["labels"])

                if config.training.gradient_accumulation:
                    loss = loss / config.training.gradient_accumulation_steps
        else:
            outputs = model(batch["features"])

            loss = loss_function(outputs, batch["labels"])

        total_loss += loss.item()

        if config.training.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if config.training.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_value)

        if config.training.gradient_accumulation:
            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        elif config.training.mixed_precision:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if config.scheduler.interval == 'step':
            if config.training.warmup_scheduler:
                if epoch >= config.training.warmup_epochs:
                    scheduler.step()
            else:
                scheduler.step()
    
    if config.training.warmup_scheduler:
        if epoch < config.training.warmup_epochs:
            scheduler.step()
        else:
            if config.scheduler.interval == 'epoch':
                scheduler.step()    
    else:
        if config.scheduler.interval == 'epoch':
            scheduler.step()

    if config.logging.prints:
        print('Learning rate:', optimizer.param_groups[0]['lr'])

    return total_loss / len(train_loader)


def validation(config, model, val_loader, loss_function):
    '''Validation loop.'''

    if config.logging.prints:
        print('Validating')

    model.eval()

    total_loss = 0.0

    preds, targets = [], []

    if config.logging.prints:
        val_loader = tqdm(val_loader)

    with torch.no_grad():
        for batch in val_loader:
            batch["features"] = [i.to(config.training.device) for i in batch["features"]]
            batch["labels"] = batch["labels"].to(config.training.device)

            outputs = model(batch["features"])

            preds.append(outputs.sigmoid().to('cpu').numpy())
            targets.append(batch["labels"].to('cpu').numpy())

            loss = loss_function(outputs, batch["labels"])
            total_loss += loss.item()

    return total_loss / len(val_loader), np.concatenate(preds), np.concatenate(targets)


def run(config, fold):
    '''Main function.'''

    # Logging
    if config.logging.wandb:
        wandb.init(project=config.logging.wandb_project_name)

    # Create working directory
    if not os.path.exists(config.paths.path_to_checkpoints):
        os.makedirs(config.paths.path_to_checkpoints)

    # Empty cache
    torch.cuda.empty_cache()

    # Get data loaders
    train_loader, val_loader = get_loaders(config, fold)

    # Get objects
    model = BasedPartyNet(**config.model.params).to(config.training.device)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    loss_function = get_loss(config)

    if config.training.mixed_precision:
        scaler = amp.GradScaler()
    else:
        scaler = None

    # Loading from checkpoint
    if config.training.resume_from_latest_checkpoint:
        try:
            cp = torch.load(os.path.join(config.paths.path_to_checkpoints, "last.pt"))
        except FileNotFoundError:
            print("The checkpoint is not found: starting a training from scratch.")
        else:
            scaler.load_state_dict(cp["scaler"])
            model.load_state_dict(cp["model"])
            optimizer.load_state_dict(cp["optimizer"])
            scheduler = get_scheduler(config, optimizer)
            for _ in range(cp["epoch"]):
                scheduler.step()

            epochs_since_improvement = cp["epochs_since_improvement"]
            current_epoch = cp["epoch"]
            best_metric = cp["best_metric"]
        
            del cp
    else:
        current_epoch = 0
        epochs_since_improvement = 0
        best_metric = 0

    # OOF
    best_preds, best_targets = None, None

    # Initializing metrics and a log
    train_losses, val_losses, metrics, learning_rates = [], [], [], []

    if config.logging.txt_file:
        with open(os.path.join(config.paths.path_to_checkpoints, "log.txt"), 'w') as file:
            file.write('Testing ' + config.general.experiment_name + ' approach\n')

    # Training

    if config.logging.prints:
        pbar = [*range(current_epoch, config.training.num_epochs)]
    else:
        pbar = tqdm(range(current_epoch, config.training.num_epochs))

    for epoch in pbar:
    
        if config.logging.prints:
            print('\nEpoch: ' + str(epoch + 1))

        start_time = time.time()

        # Train and validation steps
        train_loss = train(config, model, train_loader, optimizer, scheduler, loss_function, epoch, scaler)
        if config.split.all_data_train:
            val_loss, predictions, targets = 0, 1, 0
            current_metric = epoch + 1
        else:
            val_loss, predictions, targets = validation(config, model, val_loader, loss_function)
            current_metric = get_metric(config, targets, predictions)

        # Save the metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        metrics.append(current_metric)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Saving model and metrics
        if current_metric > best_metric:
            if config.logging.prints:
                print('New Record!')

            epochs_since_improvement = 0
            best_metric = current_metric
            best_preds = predictions
            best_targets = targets
            
            if config.training.save_best:
                save_model(config, model, epoch + 1, current_metric, optimizer, epochs_since_improvement, 
                        'best.pt', scheduler, scaler, best_metric)
        else:
            epochs_since_improvement += 1

        if config.training.save_last:
            save_model(config, model, epoch + 1, current_metric, optimizer, epochs_since_improvement,
                        'last.pt', scheduler, scaler, best_metric)

        # Logging
        if not config.logging.prints:
            pbar.set_postfix(score=current_metric, best=best_metric)
        
        if config.logging.prints:
            t = int(time.time() - start_time)
            print_report(t, train_loss, val_loss, current_metric, best_metric)

        if config.logging.txt_file:
            save_log(os.path.join(config.paths.path_to_checkpoints, "log.txt"), epoch + 1,
                    train_loss ,val_loss, best_metric)

        if epochs_since_improvement == config.training.early_stopping_epochs:
            print('Training has been interrupted by early stopping.')
            break
        
        # Empty cache
        torch.cuda.empty_cache()
        gc.collect()

    # Draw plots if needed
    if config.training.verbose_plots:
        draw_plots(train_losses, val_losses, metrics, learning_rates)
    
    return best_metric, best_preds, best_targets