import argparse
import matplotlib
import matplotlib.pyplot as plt
import pylab
# %matplotlib inline

import numpy as np

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Paraphraser')
    parser.add_argument('-m', '--model-name', default='', help='name of model to save (default: "")')
    args = parser.parse_args()

    ce_result_valid = list(np.load(f'logs/{args.model_name}/ce_result_valid.npy'))
    ce_result_train = list(np.load(f'logs/{args.model_name}/ce_result_train.npy'))
    kld_result_valid = list(np.load(f'logs/{args.model_name}/kld_result_valid.npy'))
    kld_result_train = list(np.load(f'logs/{args.model_name}/kld_result_train.npy'))

    if 'tpl' in args.model_name.lower():
        ce2_result_valid = list(np.load(f'logs/{args.model_name}/ce2_result_valid.npy'))
        ce2_result_train = list(np.load(f'logs/{args.model_name}/ce2_result_train.npy'))
    if 'gan' in args.model_name.lower():
        ce2_result_valid = list(np.load(f'logs/{args.model_name}/ce2_result_valid.npy'))
        ce2_result_train = list(np.load(f'logs/{args.model_name}/ce2_result_train.npy'))
        dg_result_valid = list(np.load(f'logs/{args.model_name}/dg_result_valid.npy'))
        dg_result_train = list(np.load(f'logs/{args.model_name}/dg_result_train.npy'))
        d_result_valid = list(np.load(f'logs/{args.model_name}/d_result_valid.npy'))
        d_result_train = list(np.load(f'logs/{args.model_name}/d_result_train.npy'))

    iter = np.arange(len(ce_result_valid)) * 500
    plt.plot(iter, ce_result_train, iter, ce_result_valid)
    plt.legend(['Train', 'Validation'])
    plt.ylabel('Cross entropy loss')
    plt.savefig('results/ce_{}.png'.format(args.model_name))
    plt.clf()

    print(np.min(ce_result_valid), np.argmin(ce_result_valid) * 500)

    plt.plot(iter, kld_result_train, iter, kld_result_valid)
    plt.legend(['Train', 'Validation'])
    plt.ylabel('KL-divergence loss')
    plt.ylim(0, 10)
    plt.savefig('results/kld_{}.png'.format(args.model_name))
    plt.clf()

    train_loss = np.sum(np.stack([ce_result_train, kld_result_train], axis=0), axis=0)
    valid_loss = np.sum(np.stack([ce_result_valid, kld_result_valid], axis=0), axis=0)
    if 'tpl' in args.model_name.lower():
        train_loss = np.sum(np.stack([ce_result_train, kld_result_train, ce2_result_train], axis=0), axis=0)
        valid_loss = np.sum(np.stack([ce_result_valid, kld_result_valid, ce2_result_valid], axis=0), axis=0)

    if 'gan' in args.model_name.lower():
        train_loss = np.sum(np.stack([ce_result_train, kld_result_train, ce2_result_train, dg_result_train], axis=0), axis=0)
        valid_loss = np.sum(np.stack([ce_result_valid, kld_result_valid, ce2_result_valid, dg_result_valid], axis=0), axis=0)

    print(np.min(valid_loss), np.argmin(valid_loss) * 500)

    plt.plot(iter, train_loss, iter, valid_loss)
    plt.legend(['Train', 'Validation'])
    plt.ylabel('Total loss')
    plt.savefig('results/total_{}.png'.format(args.model_name))
    plt.clf()
