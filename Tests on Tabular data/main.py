import sys
sys.path.append('../')
sys.path.append('./')

import argparse
import numpy as np
import torch
import pickle

from training import loop_eval_train, test
from utils import save_handler, set_seed, unset_seed, Tau_linear_decreasing_strategy, Tau_exponential_decreasing_strategy
from NPS_based_model import Model


def parse_boolean(value):
    value = value.lower()
    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False
    return False

def parse_str_with_None(value):
    value = str(value)
    return None if value == 'None' else value

def parse_int_with_none(value):
    value = str(value)
    return None if ((value == 'None') or (int(value) <= 0)) else int(value)

def get_args(str_args, dic_type=None):
    if (str_args == '') or (str_args == "''") or (str_args == 'None'):
        return []
    sep = '=' if('=' in str_args) else ':' if (':' in str_args) else None
    if sep:
        args = dict(map(lambda e: map(str, e.split(sep)), str_args.split(',')))
        return args if dic_type is None else {k: dic_type[k](args[k]) for k in args}
    else:
        args = list(map(str, str_args.split(',')))
        return args if dic_type is None else [t(e) for t, e in zip(dic_type.values(), args)]

def get_strat(param):
        if param is not None:
            split = param.split('(')
            strat = split[0]
            start, stop, step = list(map(float, split[1][:-1].split(',')))
            if strat == 'lin_dec':
                return Tau_linear_decreasing_strategy(start, stop, step)
            elif strat == 'exp_dec':
                return Tau_exponential_decreasing_strategy(start, stop, step)
        return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NPS Hyperparameters')

    parser.add_argument('--use_autoencoder', default=False, type=parse_boolean,
                        help='Whether to use the autoencoder during training')
    parser.add_argument('--n_slots', default=4, type=int,
                        help='Number of slots to use (in total)')
    parser.add_argument('--n_iter', default=3, type=int,
                        help='Number of iterations to use')
    parser.add_argument('--n_rules', default=9, type=int,
                        help='Number of rules to use (in total)')
    parser.add_argument('--sp_dim', default=8, type=parse_int_with_none,
                        help='Dimension of the primary slot')
    parser.add_argument('--s_dim', default=32, type=int,
                        help='Dimension of each slot')
    parser.add_argument('--pos_dim', default=12, type=int,
                        help='Dimension of the embeddings of each contextual slot')
    parser.add_argument('--r_dim', default=12, type=int,
                        help='Dimension of the rule embedding (for each rule)')
    parser.add_argument('--rule_mlp_dim', default=32, type=int,
                        help='Dimension of the hidden layer for each MLP (of each rule)')
    parser.add_argument('--use_null_rule', default=False, type=parse_boolean,
                        help='Whether to use a null rule (would be the last rule)')
    parser.add_argument('--dim_attn_r', default=32, type=int,
                        help='Dimension of the projection space for the rule selection attention')
    parser.add_argument('--dim_attn_c', default=64, type=int,
                        help='Dimension of the projection space for the contextual slot selection attention')
    parser.add_argument('--dim_key_r', default=None, type=parse_int_with_none,
                        help='3rd dimension of the key attention for rule selection (None or num_slots-1 or 1)')
    parser.add_argument('--dim_query_r', default=None, type=parse_int_with_none,
                        help='3rd dimension of the query attention for rule selection (None or num_rules)')
    parser.add_argument('--dim_key_c', default=None, type=parse_int_with_none,
                        help='3rd dimension of the key attention contextual slot selection (None or num_slots-1 or 1)')
    parser.add_argument('--dim_query_c', default=None, type=parse_int_with_none,
                        help='3rd dimension of the query attention contextual slot selection (None or 1)')
    
    parser.add_argument('--tau_train_r', default=1., type=float,
                        help='Temperature of the Gumbel Softmax for the rule selection during training')
    parser.add_argument('--tau_eval_r', default=0., type=float,
                        help='Temperature of the Gumbel Softmax for the rule selection during evaluation')
    parser.add_argument('--tau_train_c', default=1., type=float,
                        help='Temperature of the Gumbel Softmax for the contextual slot selection during training')
    parser.add_argument('--tau_eval_c', default=0., type=float,
                        help='Temperature of the Gumbel Softmax for the contextual slot selection during evaluation')
    parser.add_argument('--tau_strat_update_train_r', default=None, type=parse_str_with_None,
                        help='Temperature evolution strategy for the rule selection during training (replace tau_train_r)') # exp_dec(50., 0.1, 0.05) # lin_dec(50., .1, 1.)
    parser.add_argument('--tau_strat_update_eval_r', default=None, type=parse_str_with_None,
                        help='Temperature evolution strategy for the rule selection during evaluation (replace tau_eval_r)')
    parser.add_argument('--tau_strat_update_train_c', default=None, type=parse_str_with_None,
                        help='Temperature evolution strategy for the contextual slot selection during training (replace tau_train_c)')
    parser.add_argument('--tau_strat_update_eval_c', default=None, type=parse_str_with_None,
                        help='Temperature evolution strategy for the contextual slot selection during evaluation (replace tau_eval_c)')
    
    parser.add_argument('--query_r', default='Sp', type=str,
                        help='Definition of the query-ies attention for the rule selection')
    parser.add_argument('--keys_c', default='Sp,POS', type=str,
                        help='Definition of the keys attention for the contextual slot selection')
    parser.add_argument('--query_c', default='R', type=str,
                        help='Definition of the query-ies attention for the contextual slot selection')
    parser.add_argument('--input_mlp', default='Sc', type=str,
                        help='Definition of the MLP inputs (for the rules)')
    
    parser.add_argument('--hard_gs_train_r', default=True, type=parse_boolean,
                        help='Whether to use a hard Gumbel Softmax for the rule selection during training')
    parser.add_argument('--hard_gs_eval_r', default=True, type=parse_boolean,
                        help='Whether to use a hard Gumbel Softmax for the rule selection during evaluation')
    parser.add_argument('--hard_gs_train_c', default=True, type=parse_boolean,
                        help='Whether to use a hard Gumbel Softmax for the contextual slot selection during training')
    parser.add_argument('--hard_gs_eval_c', default=True, type=parse_boolean,
                        help='Whether to use a hard Gumbel Softmax for the contextual slot selection during evaluation')
    parser.add_argument('--tpr_order', default=2, type=int,
                        help='Using TPR order (value in [None, 0, 2, 3])')
    parser.add_argument('--use_mlp_rules', default=True, type=parse_boolean,
                        help='Whether to use a MLP for the rules')
    parser.add_argument('--scores_constraint_pos_visited', default=False, type=parse_boolean,
                        help='Whether to use a constraint over the usage of positional encoding of contextual slots to don\'t the same positional encoding in the next iterations')
    parser.add_argument('--use_pos_onehot', default=False, type=parse_boolean,
                        help='Whehter to use onehot encoding for the positional encoding of contextual slots (else use an learned embedding)')
    parser.add_argument('--simplified', default=False, type=parse_boolean,
                        help='Using the simplified architecture')
    parser.add_argument('--reversed_attn', default=False, type=parse_boolean,
                        help='Using the reversed attention architecture')
    parser.add_argument('--stddev_noise_r', default=0., type=float,
                        help='Level of noise to add on the attention for the rule selection')
    parser.add_argument('--stddev_noise_c', default=0., type=float,
                        help='Level of noise to add on the attention for the contextual slot selection')
    parser.add_argument('--lr', default=.0001, type=float,
                        help='Learning rate')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs')
    parser.add_argument('--bs', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--seed', default=0, type=int,
                        help='Seed')
    parser.add_argument('--use_entropy', default=False, type=parse_boolean,
                        help='Whether to use the entropy in the loss')
    parser.add_argument('--replace_mode', default=False, type=parse_boolean,
                        help='Whether to use the replacing mode (using adding mode otherwise)')
    parser.add_argument('--force_gpu', default=False, type=parse_boolean,
                        help='Whether to force the usage of GPU')
    
    parser.add_argument('--train_size', default=10_000, type=int,
                        help='Size of the training dataset')
    parser.add_argument('--test_size', default=5_000, type=int, # test_eval_size//2 = test_size = eval_size
                        help='Size of the test dataset')
    parser.add_argument('--k_fold', default=1, type=int,
                        help='Number of fold in the training dataset')
    parser.add_argument('--do_test', default=True, type=parse_boolean,
                        help='Whether to do a test at the end of the training phase')
    parser.add_argument('--data_path', default='./data/', type=str,
                        help='Data path (relative or absolute path)')
    parser.add_argument('--data_shuffle', default=True, type=parse_boolean,
                        help='Whether to shuffle the data')
    parser.add_argument('--n_classes', default=10, type=int,
                        help='The number of classes in the tasks')
    
    parser.add_argument('--save', default=True, type=parse_boolean,
                        help='Whether to save the statistics of the training/evaluation and the model properties')
    parser.add_argument('--save_path', default='./', type=str,
                        help='The path of the statistics to save')
    parser.add_argument('--save_name', default='TASK', type=str,
                        help='The name of the file to save')
    parser.add_argument('--save_model', default=False, type=parse_boolean,
                        help='Whether to save the model with the statistics')

    parser.add_argument('--encoder_callback_name', default=None, type=parse_str_with_None,
                        help='The name of the class of the encoder callback in "tables_utils" (None if no encoder)')
    parser.add_argument('--encoder_callback_args', default=None, type=parse_str_with_None,
                        help='The arguments to give in the constructor of the encoder callback class')
    parser.add_argument('--autoencoder_module_name', default='CNN_Encoder_Decoder_MNIST', type=parse_str_with_None,
                        help='The name of the file where there are the encoder and decoder definition (functions named: "get_encoder" and "get_decoder")')
    parser.add_argument('--dataloader_module_name', default='MNIST_Slot_Dataset', type=str,
                        help='The name of the file to load the datasets (should contain a function called "get_dataloaders")')
    parser.add_argument('--dataloader_sup_args', default=None, type=parse_str_with_None,
                        help='The arguments to give at the function "get_dataloaders" from the module "dataloader_module_name"')

    args = parser.parse_args()
    
    dataloaders_callback = getattr(__import__(args.dataloader_module_name), 'get_dataloaders')
    dataloaders_sup_args = {} if args.dataloader_sup_args is None else get_args(args.dataloader_sup_args, {'noise': float, 'data_path': str})
    n_classes = args.n_classes

    if args.encoder_callback_name is None:
        encoder_callback = None
    else:
        #encoder_callback_args = {} if args.encoder_callback_args is None else get_args(args.encoder_callback_args, {'stride': int, 'size': int})
        class_callback = getattr(__import__('tables_utils'), args.encoder_callback_name)
        #if isinstance(encoder_callback_args, dict):
        #    encoder_callback = class_callback(**encoder_callback_args)
        #else:
        #    encoder_callback = class_callback(*encoder_callback_args)
        encoder_callback = class_callback()
    
    device = torch.device('cuda') if args.force_gpu else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    query_r = get_args(args.query_r)
    keys_c = get_args(args.keys_c)
    query_c = get_args(args.query_c)
    input_mlp = get_args(args.input_mlp)
    
    tau_strat_update_train_r = get_strat(args.tau_strat_update_train_r)
    tau_strat_update_eval_r = get_strat(args.tau_strat_update_eval_r)
    tau_strat_update_train_c = get_strat(args.tau_strat_update_train_c)
    tau_strat_update_eval_c = get_strat(args.tau_strat_update_eval_c)
    
    set_seed(args.seed) # Use seed only for data generation

    # Load data
    if isinstance(dataloaders_sup_args, dict):
        train_eval_loaders, test_loader = dataloaders_callback(data_path=args.data_path, batch_size=args.bs, max_train_size=args.train_size, max_test_size=args.test_size, shuffle=args.data_shuffle, k_fold=args.k_fold, **dataloaders_sup_args)
    else:
        train_eval_loaders, test_loader = dataloaders_callback(data_path=args.data_path, batch_size=args.bs, max_train_size=args.train_size, max_test_size=args.test_size, shuffle=args.data_shuffle, k_fold=args.k_fold, *dataloaders_sup_args)
    
    train_losses, eval_losses, train_accuracy, eval_accuracy, train_activations, eval_activations = [], [], [], [], [], []
    best_acc, best_params = 0, None
    for k, (train_loader, eval_loader) in enumerate(train_eval_loaders):
        print(f'=== K-fold {k+1} ===')

        unset_seed()
        encoder = __import__(args.autoencoder_module_name).get_encoder(args.s_dim, args.n_slots, device)
        decoder = __import__(args.autoencoder_module_name).get_decoder(args.s_dim, args.n_slots, device)

        with torch.autograd.set_detect_anomaly(True):
            model = Model(encoder=encoder, decoder=decoder, n_classes=n_classes, n_iter=args.n_iter, n_slots=args.n_slots, s_dim=args.s_dim, sp_dim=args.sp_dim,
                          n_rules=args.n_rules, r_dim=args.r_dim, rule_mlp_hidden_dim=args.rule_mlp_dim, pos_dim=args.pos_dim, use_null_rule=args.use_null_rule,
                          transform_input_callback=encoder_callback, query_r=query_r, keys_c=keys_c, query_c=query_c, input_mlp=input_mlp,
                          stddev_noise_r=args.stddev_noise_r, stddev_noise_c=args.stddev_noise_c,# dropout=DROPOUT,
                          use_entropy=args.use_entropy, dim_attn_r=args.dim_attn_r, dim_attn_c=args.dim_attn_c,
                          dim_key_r=args.dim_key_r, dim_query_r=args.dim_query_r, dim_key_c=args.dim_key_c, dim_query_c=args.dim_query_c,
                          tau_train_r=args.tau_train_r, tau_eval_r=args.tau_eval_r, tau_train_c=args.tau_train_c, tau_eval_c=args.tau_eval_c,
                          hard_gs_train_r=args.hard_gs_train_r, hard_gs_eval_r=args.hard_gs_eval_r,
                          hard_gs_train_c=args.hard_gs_train_c, hard_gs_eval_c=args.hard_gs_eval_c,
                          replace_mode=args.replace_mode, tpr_order=args.tpr_order, simplified=args.simplified, reversed_archtiecture=args.reversed_attn,
                          scores_constraint_pos_visited=args.scores_constraint_pos_visited, use_pos_onehot=args.use_pos_onehot, device=device).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = torch.nn.BCELoss()
            
            train_loss, eval_loss, train_acc, eval_acc, \
            train_act, eval_act = loop_eval_train(train_loader, eval_loader, model, optimizer, criterion, args.epochs, device, args.use_entropy,
                                                  tau_strat_update_train_r, tau_strat_update_eval_r, tau_strat_update_train_c, tau_strat_update_eval_c,
                                                  args.use_autoencoder, encoder_callback=encoder_callback)
            
            if max(eval_acc) > best_acc:
                best_acc = max(eval_acc)
                best_params = model.state_dict()
                train_activations[:] = train_act
                eval_activations[:] = eval_acc
            train_losses.append(train_loss)
            eval_losses.append(eval_loss)
            train_accuracy.append(train_acc)
            eval_accuracy.append(eval_acc)
        print()
    
    if args.do_test or args.save_model:
        model.load_state_dict(best_params) # Get the best model from best params
    if args.do_test:
        test_loss, test_acc = test(model, criterion, test_loader, device)
    else:
        test_acc, test_loss = None, None
    
    if args.save:
        filename = save_handler(args.save_path, args.save_name)
        with open(filename, 'wb') as f:
            pickle.dump({
                'params': {
                    'use_autoencoder': args.use_autoencoder,
                    'encoder_callback': None,
                    'n_slots': args.n_slots,
                    'n_iter': args.n_iter,
                    'n_rules': args.n_rules,
                    's_dim': args.s_dim,
                    'sp_dim': args.sp_dim,
                    'pos_dim': args.pos_dim,
                    'r_dim': args.r_dim,
                    'rule_mlp_dim': args.rule_mlp_dim,
                    'use_null_rule': args.use_null_rule,
                    'dim_attn_r': args.dim_attn_r,
                    'dim_attn_c': args.dim_attn_c,
                    'query_r': ','.join(query_r) if query_r else None,
                    'keys_c': ','.join(keys_c) if keys_c else None,
                    'query_c': ','.join(query_c) if query_c else None,
                    'input_mlp': ','.join(input_mlp) if input_mlp else None,
                    'dim_key_pr': args.dim_key_r,
                    'dim_query_pr': args.dim_query_r,
                    'dim_key_c': args.dim_key_c,
                    'dim_query_c': args.dim_query_c,
                    'tau_strat_update_train_r': tau_strat_update_train_r.__name__ if tau_strat_update_train_r else None,
                    'tau_strat_update_eval_r': tau_strat_update_eval_r.__name__ if tau_strat_update_eval_r else None,
                    'tau_strat_update_train_c': tau_strat_update_train_c.__name__ if tau_strat_update_train_c else None,
                    'tau_strat_update_eval_c': tau_strat_update_eval_c.__name__ if tau_strat_update_eval_c else None,
                    'tau_train_r': args.tau_train_r,
                    'tau_eval_r': args.tau_eval_r,
                    'tau_train_c': args.tau_train_c,
                    'tau_eval_c': args.tau_eval_c,
                    'stddev_noise_r': args.stddev_noise_r,
                    'stddev_noise_c': args.stddev_noise_c,
                    'hard_gs_train_r': args.hard_gs_train_r,
                    'hard_gs_eval_p': args.hard_gs_eval_r,
                    'hard_gs_train_c': args.hard_gs_train_c,
                    'hard_gs_eval_c': args.hard_gs_eval_c,
                    'tpr_order': args.tpr_order,
                    'simplified': args.simplified,
                    'reversed_attn': args.reversed_attn,
                    'scores_constraint_pos_visited': args.scores_constraint_pos_visited,
                    'use_pos_onehot': args.use_pos_onehot,
                    # 'dropout': dropout,
                    'learning_rate': args.lr,
                    'epochs': args.epochs,
                    'batch_size': args.bs,
                    'seed': args.seed,
                    'use_entropy': args.use_entropy,
                    'num_classes': args.n_classes,
                    'num_train_size': args.train_size,
                    'num_test_size': args.test_size,
                    'k_fold': args.k_fold,
                    'device_type': device.type,
                },
                'model': model if args.save_model else None,
                'train_activations': train_activations,
                'train_loss': train_losses,
                'train_acc': train_accuracy,
                'eval_activations': eval_activations,
                'eval_loss': eval_losses,
                'eval_acc': eval_accuracy,
                'test_acc': test_acc,
                'test_loss': test_loss,
            }, f)
        print('Stats saved!')
