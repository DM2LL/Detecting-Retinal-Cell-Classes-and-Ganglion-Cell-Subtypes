import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import random
import math
seed=666
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.set_num_threads(2)

import learning_rate
import loss_functions
from utils import *
from networks import *


# Training settings
parser = argparse.ArgumentParser(description='DA')
parser.add_argument('--method', type=str, default='MinMax')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--embedding_size', type=int, default=256, help='embedding_size')
parser.add_argument('--source_name', type=str, default='TM_baron_mouse')
parser.add_argument('--target_name', type=str, default='segerstolpe_human')
parser.add_argument('--result_path', type=str, default='Results/')
parser.add_argument('--dataset_path', type=str, default='dataset/Processed-data_Pancreas/')
parser.add_argument('--num_iterations', type=int, default=15000, help="num_iterations")
parser.add_argument('--BNM_coeff', type=float, default=0.2, help="regularization coefficient for BNM loss")
parser.add_argument('--lambda_zero', type=float, default=1.0, help="regularization coefficient for MinMax loss")
parser.add_argument('--pseudo_th', type=float, default=0.0, help='pseudo_th')
parser.add_argument('--cell_th', type=int, default=20, help='cell_th')
parser.add_argument('--epoch_th', type=int, default=15000, help='epoch_th')
parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient for VAT loss')
parser.add_argument('--xi', type=float, default=10.0, metavar='XI',
                        help='hyperparameter of VAT')
parser.add_argument('--eps', type=float, default=1.0, metavar='EPS',
                        help='hyperparameter of VAT')
parser.add_argument('--ip', type=int, default=1, metavar='IP',
                        help='hyperparameter of VAT')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
kwargs = {'num_workers': 1, 'pin_memory': True}


def preprocess(args):
    dataset_path = args.dataset_path  
    print("dataset_path: ", dataset_path)
    datacounts = pd.read_csv(dataset_path + 'Merged_Expression.csv')
    labels = pd.read_csv(dataset_path + 'Merged_Labels.csv')
    Labels_of_Domains = pd.read_csv(dataset_path + 'Labels_of_Domains.csv')
    data_set = {'datacounts': datacounts.T.values, 'labels': labels.iloc[:, 0].values, 'Labels_of_Domains': Labels_of_Domains.iloc[:, 0].values}
    return data_set


def ScDA(args, data_set):
    batch_size = args.batch_size
    source_name = args.source_name 
    target_name = args.target_name 
    domain_to_indices = np.where(data_set['Labels_of_Domains'] == source_name)[0]
    print(domain_to_indices)
    train_set = {'datacounts': data_set['datacounts'][domain_to_indices], 'labels': data_set['labels'][domain_to_indices],
                 'Labels_of_Domains': data_set['Labels_of_Domains'][domain_to_indices]}
    domain_to_indices = np.where(data_set['Labels_of_Domains'] == target_name)[0]
    print(domain_to_indices)
    test_set = {'datacounts': data_set['datacounts'][domain_to_indices], 'labels': data_set['labels'][domain_to_indices],
                'Labels_of_Domains': data_set['Labels_of_Domains'][domain_to_indices]}
    print('source labels:', np.unique(train_set['labels']), 'target labels:', np.unique(test_set['labels']))
    test_set_eval = {'datacounts': data_set['datacounts'][domain_to_indices], 'labels': data_set['labels'][domain_to_indices],
                     'Labels_of_Domains': data_set['Labels_of_Domains'][domain_to_indices]}
    print(train_set['datacounts'].shape, test_set['datacounts'].shape)

    data = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_set['datacounts']), torch.LongTensor(matrix_one_hot(train_set['labels'], int(max(train_set['labels'])+1)).long()))
    source_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    data = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_set['datacounts']), torch.LongTensor(matrix_one_hot(test_set['labels'], int(max(train_set['labels'])+1)).long()))
    target_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    
    class_num = max(train_set['labels']) + 1
    class_num_test = max(test_set['labels']) + 1

    ## re-weighting the classifier
    cls_num_list = [np.sum(train_set['labels'] == i) for i in range(class_num)]
    # Normalized weights based on inverse number of effective data per class.
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

    ## set base network
    embedding_size = args.embedding_size
    base_network = FeatureExtractor(num_inputs=train_set['datacounts'].shape[1], embed_size = embedding_size).cuda()
    label_predictor = LabelPredictor(base_network.output_num(), class_num).cuda()
    total_model = nn.Sequential(base_network, label_predictor)

    print("output size of FeatureExtractor and LabelPredictor: ", base_network.output_num(), class_num)
    ad_net = scAdversarialNetwork(base_network.output_num(), 1024).cuda()

    ## set optimizer
    config_optimizer = {"lr_type": "inv", "lr_param": {"lr": 0.001, "gamma": 0.001, "power": 0.75}}
    parameter_list = base_network.get_parameters() + ad_net.get_parameters() + label_predictor.get_parameters()
    optimizer = optim.SGD(parameter_list, lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True)
    schedule_param = config_optimizer["lr_param"]
    lr_scheduler = learning_rate.schedule_dict[config_optimizer["lr_type"]]

    ## train
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    epoch_global = 0.0
    hit = False
    
    for epoch in range(args.num_iterations):
        if epoch % (2500) == 0 and epoch != 0:
            feature_target = base_network(torch.FloatTensor(test_set['datacounts']).cuda())
            output_target = label_predictor.forward(feature_target)
            softmax_out = nn.Softmax(dim=1)(output_target)
            predict_prob_arr, predict_label_arr = torch.max(softmax_out, 1)
            if epoch == args.epoch_th:
                data = torch.utils.data.TensorDataset(torch.FloatTensor(test_set['datacounts']), predict_label_arr.cpu())
                target_loader_align = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
            result_path = args.result_path #"results/"
            model_file = result_path + 'final_model_' + str(epoch) + source_name + target_name+'.ckpt'
            torch.save({'base_network': base_network.state_dict(), 'label_predictor': label_predictor.state_dict()}, model_file)

            if not os.path.exists(result_path):
                os.makedirs(result_path)
            with torch.no_grad():
                code_arr_s = base_network(Variable(torch.FloatTensor(train_set['datacounts']).cuda()))
                code_arr_t = base_network(Variable(torch.FloatTensor(test_set_eval['datacounts']).cuda()))
                code_arr = np.concatenate((code_arr_s.cpu().data.numpy(), code_arr_t.cpu().data.numpy()), 0)

            Labels_dict = pd.read_csv(args.dataset_path + 'Labels_dict.csv')
            Labels_dict = pd.DataFrame(zip(Labels_dict.iloc[:,0], Labels_dict.index), columns=['digit','label'])
            Labels_dict = Labels_dict.to_dict()['label']
            # transform digit label to cell type name
            y_pred_label = [Labels_dict[x] if x in Labels_dict else x for x in predict_label_arr.cpu().data.numpy()]

            pred_labels_file = result_path + 'pred_labels_' + source_name + "_" + target_name + "_" + str(epoch) + ".csv"
            pd.DataFrame([predict_prob_arr.cpu().data.numpy(), y_pred_label],  index=["pred_probability", "pred_label"]).to_csv(pred_labels_file, sep=',')
            embedding_file = result_path + 'embeddings_' + source_name + "_" + target_name + "_" + str(epoch)+ ".csv"
            pd.DataFrame(code_arr).to_csv(embedding_file, sep=',')

            ## only for evaluation
            acc_by_label = np.zeros( class_num_test )
            all_label = test_set['labels']
            for i in range(class_num_test):
                 acc_by_label[i] = np.sum(predict_label_arr.cpu().data.numpy()[all_label == i] == i) / np.sum(all_label == i)
            np.set_printoptions(suppress=True)
            print('iter:', epoch, "average acc over all test cell types: ", round(np.nanmean(acc_by_label), 3))
            print("acc of each test cell type: ", acc_by_label)

        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        label_predictor.train(True)

        optimizer = lr_scheduler(optimizer, epoch, **schedule_param)
        optimizer.zero_grad()
        
        if epoch % len_train_source == 0:
            iter_source = iter(source_loader)
            epoch_global = epoch_global + 1
        if epoch % len_train_target == 0:
            if epoch < args.epoch_th:
                iter_target = iter(target_loader)
            else:
                hit = True
                iter_target = iter(target_loader_align)
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source, labels_target = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda(), labels_target.cuda()

        feature_source = base_network(inputs_source)
        feature_target = base_network(inputs_target)
        features = torch.cat((feature_source, feature_target), dim=0)

        output_source = label_predictor.forward(feature_source)
        output_target = label_predictor.forward(feature_target)

        ######## VAT and BNM loss
        # LDS should be calculated before the forward for cross entropy
        vat_loss = loss_functions.VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)
        lds_loss = vat_loss(total_model, inputs_target)

        softmax_tgt = nn.Softmax(dim=1)(output_target[:, 0:class_num])
        _, s_tgt, _ = torch.svd(softmax_tgt)
        BNM_loss = -torch.mean(s_tgt)

        ####### MinMax loss
        if args.method == 'MinMax':
            domain_prob_discriminator_1_source = ad_net.forward(feature_source)
            domain_prob_discriminator_1_target = ad_net.forward(feature_target)
            adv_loss = loss_functions.BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), \
                                                     predict_prob=domain_prob_discriminator_1_source)
            adv_loss += loss_functions.BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), \
                                                      predict_prob=1 - domain_prob_discriminator_1_target)
            MinMax_loss = adv_loss

        ###### CrossEntropyLoss
        classifier_loss = nn.CrossEntropyLoss(weight=per_cls_weights)(output_source, torch.max(labels_source, dim=1)[1])

        ###### cmmdLoss
        epoch_th = args.epoch_th
        if epoch < args.epoch_th or hit == False:
            cmmd_loss = torch.FloatTensor([0.0]).cuda()
            pass
        elif hit == True:
            base = 1.0  # sigma for MMD
            sigma_list = [1, 2, 4, 8, 16]
            sigma_list = [sigma / base for sigma in sigma_list]
            cmmd_loss = loss_functions.mix_rbf_mmd2(feature_source, feature_target, sigma_list)

        if epoch > epoch_th:
            lds_loss = torch.FloatTensor([0.0]).cuda()
        if epoch <= args.num_iterations:
            progress = epoch / args.epoch_th #args.num_iterations
        else:
            progress = 1

        lambd = 2 / (1 + math.exp(-10 * progress)) - 1
        total_loss = classifier_loss + lambd*args.lambda_zero * MinMax_loss + lambd*args.BNM_coeff*BNM_loss + lambd*args.alpha*lds_loss + lambd*cmmd_loss    
        total_loss.backward()
        optimizer.step()


if __name__ == "__main__":
    data_set = preprocess(args)
    ScDA(args, data_set=data_set)