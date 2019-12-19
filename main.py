import argparse
import random
import numpy as np
import time
import torch
from torch import optim
from lf_evaluator import *
from models import *
from data import *
from utils import *
from typing import List

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='main.py')
    
    # General system running and configuration options
    parser.add_argument('--do_nearest_neighbor', dest='do_nearest_neighbor', default=False, action='store_true', help='run the nearest neighbor model')

    parser.add_argument('--train_path', type=str, default='data/geo_train.tsv', help='path to train data')
    parser.add_argument('--dev_path', type=str, default='data/geo_dev.tsv', help='path to dev data')
    parser.add_argument('--test_path', type=str, default='data/geo_test.tsv', help='path to blind test data')
    parser.add_argument('--test_output_path', type=str, default='geo_test_output.tsv', help='path to write blind test results')
    parser.add_argument('--domain', type=str, default='geo', help='domain (geo for geoquery)')
    
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=100, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes
    args = parser.parse_args()
    return args


class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """
    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability,
        and a tokenized input string. If you're just doing one-best decoding of example ex and you
        produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # N.B. a list!
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs


class Seq2SeqSemanticParser(object):
    def __init__(self, derivative_derived):
        self.derivative = derivative_derived
        #raise Exception("implement me!")
        # Add any args you need here

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        return self.derivative
        raise Exception("implement me!")


class Seq2SeqSemanticParser_evaluate(object):
    def __init__(self, encoder_net, decoder_net, hidden_size, input_indexer, output_indexer):
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.hidden_size = hidden_size
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        #raise Exception("implement me!")
        # Add any args you need here

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
        input_max_length = 19
        output_max_length = 65
        all_test_input_data = make_padded_input_tensor(test_data, self.input_indexer, input_max_length, reverse_input=False)
        all_test_output_data = make_padded_output_tensor(test_data, self.output_indexer, output_max_length)

        from torch.utils.data import TensorDataset, DataLoader
        test_data_tensor = TensorDataset(torch.from_numpy(all_test_input_data), torch.from_numpy(all_test_output_data))
        batch_size = 1
        test_loader = DataLoader(test_data_tensor, shuffle=False, batch_size=batch_size)

        test_derive = []
        counter = 0

        for inputs, labels in test_loader:
            decoded_words = []
            with torch.no_grad():
                encoder_hidden = self.encoder_net.initHidden()
                encoder_outputs = torch.zeros(input_max_length, self.hidden_size)#, device=device)
                for ei in range(input_max_length):
                    encoder_output, encoder_hidden = self.encoder_net(inputs[0][ei],encoder_hidden)
                    encoder_outputs[ei] += encoder_output[0, 0]
                SOS_token = 1
                EOS_token = 2
                PAD_token = 0

                decoder_input = torch.tensor([[SOS_token]])#, device=device)
                decoder_hidden = encoder_hidden
                decoded_words = []

                for di in range(output_max_length):
                    decoder_output, decoder_hidden = self.decoder_net(
                    decoder_input, decoder_hidden)
                    topv, topi = decoder_output.data.topk(1)
                    if topi.item() == EOS_token or topi.item() == PAD_token:
                        #decoded_words.append('<EOS>')
                        break
                    else:
                        #decoded_words.append(output_lang.index2word[topi.item()])
                        decoded_words.append(self.output_indexer.get_object(topi.item()))

                    decoder_input = topi.squeeze().detach()
        
            test_derive.append([Derivation(test_data[counter], 1.0, decoded_words)])
            counter +=1
        del self.encoder_net
        del self.decoder_net
        
        return test_derive
        raise Exception("implement me!")

def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int, reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])


def encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb: EmbeddingLayer, model_enc: RNNEncoder):
    """
    Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
    inp_lens_tensor lengths.
    YOU DO NOT NEED TO USE THIS FUNCTION. It's merely meant to illustrate the usage of EmbeddingLayer and RNNEncoder
    as they're given to you, as well as show what kinds of inputs/outputs you need from your encoding phase.
    :param x_tensor: [batch size, sent len] tensor of input token indices
    :param inp_lens_tensor: [batch size] vector containing the length of each sentence in the batch
    :param model_input_emb: EmbeddingLayer
    :param model_enc: RNNEncoder
    :return: the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting which words
    are real and which ones are pad tokens), and the encoder final states (h and c tuple)
    E.g., calling this with x_tensor (0 is pad token):
    [[12, 25, 0, 0],
    [1, 2, 3, 0],
    [2, 0, 0, 0]]
    inp_lens = [2, 3, 1]
    will return outputs with the following shape:
    enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
    enc_final_states = 3 x dim
    """
    input_emb = model_input_emb.forward(x_tensor)
    (enc_output_each_word, enc_context_mask, enc_final_states) = model_enc.forward(input_emb, inp_lens_tensor)
    enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
    return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)


def train_model_encdec(train_data: List[Example], test_data: List[Example], input_indexer, output_indexer, args) -> Seq2SeqSemanticParser:
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param test_data:
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, reverse_input=False)
    all_test_input_data = make_padded_input_tensor(test_data, input_indexer, input_max_len, reverse_input=False)

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(test_data, output_indexer, output_max_len)

    print("Train length: %i" % input_max_len)
    print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
    print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # First create a model. Then loop over epochs, loop over examples, and given some indexed words, call
    # the encoder, call your decoder, accumulate losses, update parameters
    from torch.utils.data import TensorDataset, DataLoader
    # create Tensor datasets
    train_data_tensor = TensorDataset(torch.from_numpy(all_train_input_data), torch.from_numpy(all_train_output_data))
    test_data_tensor = TensorDataset(torch.from_numpy(all_test_input_data), torch.from_numpy(all_test_output_data))
    #print(train_data_tensor[0])
    #print(train_data_tensor[0][0].shape)

    # dataloaders
    batch_size = 1
    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data_tensor, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_data_tensor, shuffle=False, batch_size=batch_size)

    # Instantiate the model w/ hyperparams
    input_size = 238
    output_size = 153 #np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
    hidden_dim = 512


    encoder_net = RNNEncoder(input_size, hidden_dim)
    decoder_net = RNNDecoder(hidden_dim, output_size)

    print("encoder_net:", encoder_net)
    print("decoder_net:", decoder_net)

    target_length = 65



    # loss and optimization functions
    lr=0.001 #0.001
    
    encoder_optimizer = torch.optim.Adam(encoder_net.parameters(), lr=lr)
    decoder_optimizer = torch.optim.Adam(decoder_net.parameters(), lr=lr)
    criterion1 = nn.NLLLoss()

    # training params

    epochs = 20 #20 

    counter = 0
    print_every = 50
    
    num_correct = 0
    
    #net.train()
    for epoch_iter in range(epochs):
        print("epoch_iter", epoch_iter)
        decoded_sentences_train = []
        train_derive = []
        for inputs, labels in train_loader:
            decoded_tokens_indexed = []


            counter += 1
            encoder_hidden = encoder_net.initHidden()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            

            encoder_outputs = torch.zeros(input_max_len, hidden_dim)
            teacher_forcing_ratio = 0.5
            loss = 0

            for ei in range(input_max_len):
                encoder_output, encoder_hidden = encoder_net(inputs[0][ei],encoder_hidden)
                # if ei == 0:
                #     encoder_output, encoder_hidden = encoder_net(inputs[0][ei], (encoder_hidden, encoder_hidden))
                # else:
                #     encoder_output, encoder_hidden = encoder_net(inputs[0][ei], (encoder_hidden[0], encoder_hidden[1]))                   
                encoder_outputs[ei] = encoder_output[0, 0]
            POS_token = 0
            SOS_token = 1
            EOS_token = 2

            decoder_input = torch.tensor([[SOS_token]])#, device=device)
            decoder_hidden = encoder_hidden

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                        # Teacher forcing: Feed the target as the next input
                        for di in range(target_length):
                            decoder_output, decoder_hidden = decoder_net(
                                decoder_input, decoder_hidden)#, encoder_outputs)
                            topv, topi = decoder_output.topk(1)
                            true_label = labels[0][di].unsqueeze(0)
                            if true_label == -1:
                                true_label = torch.tensor(0).unsqueeze(0)
                                print("changed")
                            #print("with teacher focing:",labels[0][di].unsqueeze(0) )
                            loss += criterion1(decoder_output, true_label)
                            decoder_input = true_label #labels[0][di]  # Teacher forcing
                            decoded_tokens_indexed.append(topi.numpy())
                            # if decoder_input.item() == EOS_token or decoder_input.item() == POS_token:
                            #     break 

            else:
                        for di in range(65):
                            decoder_output, decoder_hidden = decoder_net(
                                decoder_input, decoder_hidden)#, encoder_outputs)
                            topv, topi = decoder_output.topk(1)
                            decoder_input = topi.squeeze().detach()  # detach from history as input
                            true_label = labels[0][di].unsqueeze(0)
                            if true_label == -1:
                                true_label = torch.tensor(0).unsqueeze(0)
                                print("changed")
                            loss += criterion1(decoder_output, true_label)
                            decoded_tokens_indexed.append(topi.numpy())
                            #print(di)
                            # if decoder_input.item() == EOS_token or decoder_input.item() == POS_token:
                            #     break                                
            loss.backward()
            decoded_tokens_indexed = np.stack(decoded_tokens_indexed,axis=0).flatten()
            decoded_sentences_train.append(decoded_tokens_indexed)
            
            

            encoder_optimizer.step()
            decoder_optimizer.step()

            loss += loss.item() / target_length
            y_token = []

            if counter % print_every == 0:
                print("ran sentences:", counter)
            eos_index = (labels == 2).nonzero()
            #print(eos_index[0,1].numpy())
            count = 0
            for x in decoded_tokens_indexed:
                if x == 0 or x == 2:
                    break
                y_token.append(output_indexer.get_object(x))
                count += 1
                if count > eos_index[0,1].numpy():
                	break
                    
                #print(y_token)
            train_derive.append([Derivation(train_data[counter-1], 1.0, y_token)])
        #print(len([Derivation(test_data[counter-1], 1.0, y_token)]))

        evaluate(train_data, Seq2SeqSemanticParser(train_derive), True)
        if (epoch_iter+1) % 2 == 0:
        	evaluate(test_data, Seq2SeqSemanticParser_evaluate(encoder_net, decoder_net, hidden_dim, input_indexer, output_indexer))
        counter = 0
    return Seq2SeqSemanticParser_evaluate(encoder_net, decoder_net, hidden_dim, input_indexer, output_indexer)
    raise Exception("Implement the rest of me to train your encoder-decoder model")


def evaluate(test_data: List[Example], decoder, train = False, example_freq=50, print_output=True, outfile=None):
    """
    Evaluates decoder against the data in test_data (could be dev data or test data). Prints some output
    every example_freq examples. Writes predictions to outfile if defined. Evaluation requires
    executing the model's predictions against the knowledge base. We pick the highest-scoring derivation for each
    example with a valid denotation (if you've provided more than one).
    :param test_data:
    :param decoder:
    :param example_freq: How often to print output
    :param print_output:
    :param outfile:
    :return:
    """
    e = GeoqueryDomain()
    pred_derivations = decoder.decode(test_data)
    if train:
    	java_crashes = True
    else:
    	java_crashes = False
    if java_crashes:
        selected_derivs = [derivs[0] for derivs in pred_derivations]
        denotation_correct = [False for derivs in pred_derivations]
    else:
        selected_derivs, denotation_correct = e.compare_answers([ex.y for ex in test_data], pred_derivations, quiet=True)
    print_evaluation_results(test_data, selected_derivs, denotation_correct, example_freq, print_output)
    # Writes to the output file if needed
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(selected_derivs[i].y_toks) + "\n")
        out.close()


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data

    train, dev, test = load_datasets(args.train_path, args.dev_path, args.test_path, domain=args.domain)
    train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = index_datasets(train, dev, test, args.decoder_len_limit)
    print("%i train exs, %i dev exs, %i input types, %i output types" % (len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))
    print("Input indexer: %s" % input_indexer)
    print("Output indexer: %s" % output_indexer)
    print("Here are some examples post tokenization and indexing:")
    for i in range(0, min(len(train_data_indexed), 10)):
        print(train_data_indexed[i])
    if args.do_nearest_neighbor:
        decoder = NearestNeighborSemanticParser(train_data_indexed)
        evaluate(dev_data_indexed, decoder)
    else:
        decoder = train_model_encdec(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)
        #evaluate(dev_data_indexed, decoder)
    print("=======FINAL EVALUATION ON BLIND TEST=======")
    evaluate(test_data_indexed, decoder, print_output=False, outfile="geo_test_output_enc_dec_512.tsv")


