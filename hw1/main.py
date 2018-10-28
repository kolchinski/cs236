import argparse

import sys
import os
import yaml
import torch
import numpy as np
import pickle as pkl
from dataset import NIPS2015Dataset
from model import RNN

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAMPLE_SEQ_LEN = 1000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Directory of saving checkpoints')
    parser.add_argument('--data_dir', type=str, default='data/', help='Directory of papers.csv')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory of putting logs')
    parser.add_argument('--gpu', action='store_true', help="Turn on GPU mode")

    args = parser.parse_args()
    return args


def dict2namespace(config):
    new_config = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(new_config, key, value)
    return new_config


def parse_config(args):
    with open('config.yml', 'r') as f:
        config = yaml.load(f)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return dict2namespace(config)

def plot_log_p(filename, dataset, rnn):
    with open(filename + '.pkl', 'rb') as f:
        lls = []
        data = pkl.load(f)
        for i, str in data.items():
            str_np = np.asarray([dataset.char2idx[c] for c in str])
            lls.append(rnn.compute_prob(str_np))

    with open(filename + '_raw.pkl', 'wb') as f:
        pkl.dump(lls, f, protocol=pkl.HIGHEST_PROTOCOL)

    plt.figure()
    plt.hist(lls)
    plt.xlabel('Log-likelihood')
    plt.xlim([-800, -50])
    plt.ylabel('Counts')
    plt.title(filename)
    plt.savefig(filename + '.png', bbox_inches='tight')
    plt.show()
    plt.close()
    print("# Figure written to %s.png." % filename)


def main():
    args = parse_args()
    config = parse_config(args)
    np.random.seed(config.seed)
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(config.seed)

    dataset = NIPS2015Dataset(batch_size=config.batch_size,
                              seq_len=config.seq_len,
                              data_folder=args.data_dir)

    rnn = RNN(
        vocab_size=dataset.voc_len,
        embedding_dim=config.embedding_dim,
        num_lstm_units=config.num_lstm_units,
        num_lstm_layers=config.num_lstm_layers,
        dataset=dataset,
        device=device
    )

    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint.pth'), map_location=device)
    rnn.load_state_dict(checkpoint['rnn'])
    print("# RNN weights restored.")

    # question 3)
    with open('samples.txt', 'w') as f:
        for i in range(5):
            text = 'sample {}: '.format(i+1)
            sample = rnn.sample(SAMPLE_SEQ_LEN)
            text += ''.join([dataset.idx2char[i] for i in sample])
            f.write(text + '\n')
    print("# Samples written to samples.txt.")

    # question 4)
    plot_log_p('random', dataset, rnn)
    plot_log_p('shakespeare', dataset, rnn)
    plot_log_p('nips', dataset, rnn)

    # question 5)
    with open('snippets.pkl', 'rb') as f:
        snippets = pkl.load(f)
    lbls = []
    for snippet in snippets:
        # Compute the log-likelihood of the current snippet
        ll = rnn.compute_prob(np.asarray([dataset.char2idx[c] for c in snippet]))
        ##### complete the code here #####
        # infer the label of the current snippet and append it to lbls.
        # If the snippet is generated randomly, append 0
        # If the snippet is from Shakespeare's work, append 1
        # If the snippet is retrieved from a NIPS paper, append 2
        ##################################
        if ll < -500: lbl = 0
        elif ll < -215: lbl = 1
        else: lbl = 2
        lbls.append(lbl)

    with open("answers.pkl", 'wb') as f:
        pkl.dump(lbls, f, protocol=pkl.HIGHEST_PROTOCOL)
    print("# Answers written to answers.pkl.")

    return 0

if __name__ == '__main__':
    sys.exit(main())
