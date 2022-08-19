import torch
import numpy as np


############### Train-Eval loop ################

def loop_eval_train(train_loader, eval_loader, model, optimizer,
                    criterion, epochs, device, use_entropy,
                    tau_strat_update_train_pr, tau_strat_update_eval_pr,
                    tau_strat_update_train_c, tau_strat_update_eval_c,
                    use_autoencoder, n_chans=1, encoder_callback=None):
    assert (use_autoencoder and encoder_callback is not None) or not use_autoencoder
    tau_strategies = [tau_strat_update_train_pr, tau_strat_update_eval_pr, tau_strat_update_train_c, tau_strat_update_eval_c]
    tau_attributes_name = ['tau_train_pr', 'tau_eval_pr', 'tau_train_c', 'tau_eval_c']
    
    # Training loop
    eval_losses = []
    eval_accuracy = []
    train_losses = []
    train_accuracy = []
    eval_activations = []
    train_activations = []

    for n in range(epochs):
        print(f'Epoch: {n+1}', end='\t')
        for tau_strat, tau_attr in zip(tau_strategies, tau_attributes_name):
            if tau_strat is not None:
                tau_strat(model, tau_attr)
        
        # TRAINING
        loss = []
        n_correct, nb = 0, 0
        list_y, list_y_hat, list_acts = [], [], []
        model.train()
        for data in train_loader:
            Xi = data[0].to(device)
            yi = data[1].to(device)
            optimizer.zero_grad()
            model.reset_arrays_record()
            yi_hat = model(Xi)
            b_loss = criterion(yi_hat, yi) + (model.entropy if use_entropy else 0)# + torch.sum(torch.abs(model.pos_visited-torch.ones(*model.pos_visited.size()).to(device)))
            if use_autoencoder:
                if encoder_callback is not None:
                    Xi_patches = encoder_callback(Xi)
                else:
                    Xi_patches = Xi.reshape((-1, n_chans, Xi.size(-2), Xi.size(-1)))
                yi_hat_patches = model.autoencoder(Xi_patches)
                b_loss += criterion(yi_hat_patches, Xi_patches)
            b_loss.backward()
            loss.append(b_loss.detach().cpu().numpy())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            yi_labels = torch.argmax(yi, dim=-1)
            yi_hat_labels = torch.argmax(yi_hat, dim=-1)
            n_correct += (yi_labels == yi_hat_labels).float().sum().item()
            nb += Xi.size(0)
            list_y += yi_labels.detach().cpu().numpy().tolist()
            list_y_hat += yi_hat_labels.detach().cpu().numpy().tolist()
            list_acts += model.get_activations().transpose(2, 0, 1).tolist()
        train_activations.append({
            'y': np.array(list_y),
            'y_hat' : np.array(list_y_hat),
            'activations': np.array(list_acts),
        })
        train_losses.append(np.mean(loss))
        train_accuracy.append(n_correct/nb)
        print('Train loss:', train_losses[-1], '\t Train acc:', train_accuracy[-1], end='\t')

        # EVALUATION
        if eval_loader is not None:
            loss = []
            n_correct, nb = 0, 0
            list_y, list_y_hat, list_acts = [], [], []
            with torch.no_grad():
                model.eval()
                for data in eval_loader:
                    Xi = data[0].to(device)
                    yi = data[1].to(device)
                    model.reset_arrays_record()
                    yi_hat = model(Xi)
                    l = criterion(yi_hat, yi)# + torch.sum(torch.abs(model.pos_visited-torch.ones(*model.pos_visited.size()).to(device)))
                    if use_autoencoder:
                        if encoder_callback is not None:
                            Xi_patches = encoder_callback(Xi)
                        else:
                            Xi_patches = Xi.reshape((-1, n_chans, Xi.size(-2), Xi.size(-1)))
                        yi_hat_patches = model.autoencoder(Xi_patches)
                        l += criterion(yi_hat_patches, Xi_patches)
                    loss.append(l.detach().cpu().numpy() + (model.entropy if use_entropy else 0))
                    yi_labels = torch.argmax(yi, dim=-1)
                    yi_hat_labels = torch.argmax(yi_hat, dim=-1)
                    n_correct += (yi_labels == yi_hat_labels).float().sum().item()
                    nb += Xi.size(0)
                    list_y += yi_labels.detach().cpu().numpy().tolist()
                    list_y_hat += yi_hat_labels.detach().cpu().numpy().tolist()
                    list_acts += model.get_activations().transpose(2, 0, 1).tolist()
                eval_activations.append({
                    'y': np.array(list_y),
                    'y_hat' : np.array(list_y_hat),
                    'activations': np.array(list_acts),
                })
            eval_losses.append(np.mean(loss))
            eval_accuracy.append(n_correct/nb)
            print('Eval loss:', eval_losses[-1], '\t Eval acc:', eval_accuracy[-1])
        else:
            print()
    
    return train_losses, eval_losses, train_accuracy, eval_accuracy, train_activations, eval_activations


def test(model, criterion, test_loader, device):
    loss = []
    n_correct = 0
    nb = 0
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(test_loader):
            Xi = data[0].to(device)
            yi = data[1].to(device)
            yi_hat = model(Xi)
            l = criterion(yi_hat, yi)
            loss.append(l.cpu().detach().numpy())
            yi_labels = torch.argmax(yi, dim=-1)
            yi_hat_labels = torch.argmax(yi_hat, dim=-1)
            n_correct += (yi_labels == yi_hat_labels).float().sum().item()
            nb += Xi.size(0)
    test_acc = n_correct/nb
    test_loss = np.mean(loss)
    print('Test loss:', test_loss, end='\t')
    print('Test accuracy:', test_acc, end='\n\n')

    return test_loss, test_acc