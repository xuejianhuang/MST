import numpy as np
import torch
from visdom import Visdom
from models import *
from data_process import *
import logging
import datetime


def macro_f1(pred, targ, num_classes=None):
    # pred = torch.max(pred, 1)[1]
    tp_out = []
    fp_out = []
    fn_out = []
    if num_classes is None:
        num_classes = sorted(set(targ.cpu().numpy().tolist()))
    else:
        num_classes = range(num_classes)
    for i in num_classes:
        tp = ((pred == i) & (targ == i)).sum().item()
        fp = ((pred == i) & (targ != i)).sum().item()
        fn = ((pred != i) & (targ == i)).sum().item()
        tp_out.append(tp)
        fp_out.append(fp)
        fn_out.append(fn)

    eval_tp = np.array(tp_out)
    eval_fp = np.array(fp_out)
    eval_fn = np.array(fn_out)

    precision = eval_tp / (eval_tp + eval_fp)
    precision[np.isnan(precision)] = 0
    precision_real = precision[0]
    precision_fake = precision[1]
    precision = np.mean(precision)

    recall = eval_tp / (eval_tp + eval_fn)
    recall[np.isnan(recall)] = 0
    recall_real = recall[0]
    recall_fake = recall[1]
    recall = np.mean(recall)

    f1 = 2 * (precision * recall) / (precision + recall)
    f1_real = 2 * (precision_real * recall_real) / (precision_real + recall_real)
    f1_fake = 2 * (precision_fake * recall_fake) / (precision_fake + recall_fake)
    return f1, precision, recall, f1_real, precision_real, recall_real, f1_fake, precision_fake, recall_fake


def accuracy(pred, targ):
    # pred = torch.max(pred, 1)[1]
    acc = ((pred == targ).float()).sum().item() / targ.size()[0]
    return acc


def saveTestResult(dataset, model_mode, test_result_dict):
    with open('result_test_{}_{}.txt'.format(dataset, model_mode), 'a', encoding='utf-8', newline='') as f:
        string1 = 'acc:' + str(test_result_dict['acc']) + '\t' + 'f1:' + str(
            test_result_dict['f1']) + '\t' + 'precision:' + str(
            test_result_dict['prec']) + '\t' + 'recall:' + str(test_result_dict['rec']) + '\n'
        string2 = 'f1_real:' + str(test_result_dict['f1_real']) + '\t' + 'precision_real:' + str(
            test_result_dict['prec_real']) + '\t' + 'recall_real:' + str(test_result_dict['rec_real']) + '\n'
        string3 = 'f1_fake:' + str(test_result_dict['f1_fake']) + '\t' + 'precision_fake:' + str(
            test_result_dict['prec_fake']) + '\t' + 'recall_fake:' + str(test_result_dict['rec_fake']) + '\n\n\n'
        f.writelines(string1)
        f.writelines(string2)
        f.writelines(string3)


def set_torch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_logger(args, log_path='../logs'):
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logger_name = f"{args.dataset}_{args.model}_{args.batch}_{time}"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    log_path = os.path.join(log_path, logger_name + '.log')
    fh = logging.FileHandler(log_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def log_args(logger, args):
    args = vars(args)
    logger.info("=" * 30 + " args " + "=" * 30)
    for k in args.keys():
        logger.info(f"{k}: {args[k]}")
    logger.info("=" * 30 + " End args " + "=" * 30)


def log_config(logger, config):
    logger.info("=" * 30 + " config " + "=" * 30)
    for atrr in dir(config):
        if not atrr.startswith('_'):
            logger.info(f"{atrr}: {getattr(config, atrr)}")
    logger.info("=" * 30 + " End config " + "=" * 30)


def log_testResult(logger, test_result_dict):
    logger.info('--------------------- test results-------------------------------')
    logger.info('acc:' + str(test_result_dict['acc']) + '  prec:' + str(test_result_dict['prec']) +
                '  rec:' + str(test_result_dict['rec']) + '  f1:' + str(test_result_dict['f1']))


def getModelAndData(config, model_name='completed', dataset='pheme'):
    model_mode = model_name
    pathset = path_Set(dataset)
    data_process = DataProcessor(config.text_max_length, pathset)
    if model_mode == 'propagation':
        train_dict, val_dict, test_dict, n_nodes, mid2bert_tokenizer = data_process.getData(model_mode, config.train,
                                                                                            config.val, config.test)
        model = propagation_DynamicGCN(n_nodes=n_nodes, mid2bert_tokenizer=mid2bert_tokenizer,
                                       bert_path=pathset.path_bert, config=config)
    elif model_mode == 'semantic':
        train_dict, val_dict, test_dict, mid2bert_tokenizer, text_embedding = data_process.getData(model_mode,
                                                                                                   config.train,
                                                                                                   config.val,
                                                                                                   config.test)
        model = semantic_DynamicGCN(text_embedding=text_embedding, mid2bert_tokenizer=mid2bert_tokenizer,
                                    bert_path=pathset.path_bert, config=config)
    elif model_mode == 'user':
        train_dict, val_dict, test_dict, idx2user_dict, mid2bert_tokenizer = data_process.getData(model_mode,
                                                                                                  config.train,
                                                                                                  config.val,
                                                                                                  config.test)
        model = user_DynamicGCN(idx2user_dict=idx2user_dict, mid2bert_tokenizer=mid2bert_tokenizer,
                                bert_path=pathset.path_bert, config=config)
    elif model_mode in ['dual', 'wotf', 'static']:
        train_dict, val_dict, test_dict, idx2user_dict, mid2bert_tokenizer, text_embedding = data_process.getData(
            model_mode, config.train, config.val, config.test)
        if model_mode == 'dual':
            model = dual_DynamicGCN(text_embedding=text_embedding, idx2user_dict=idx2user_dict,
                                    mid2bert_tokenizer=mid2bert_tokenizer, bert_path=pathset.path_bert, config=config)
        elif model_mode == 'wotf':
            model = dual_DynamicGCN_wotf(text_embedding=text_embedding, idx2user_dict=idx2user_dict,
                                         mid2bert_tokenizer=mid2bert_tokenizer, bert_path=pathset.path_bert, config=config)
        elif model_mode == 'static':
            model = dual_StaticGCN(text_embedding=text_embedding, idx2user_dict=idx2user_dict,
                                   mid2bert_tokenizer=mid2bert_tokenizer, bert_path=pathset.path_bert, config=config)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model, train_dict, val_dict, test_dict


class Visualize():
    def __init__(self, server, host):
        self.server = server
        self.host = host
        self.viz = Visdom(server=self.server, port=self.host)

    def append(self, y, x, name, win, epoch):
        if epoch == 0 and name in ['train_loss', 'train_acc']:
            self.viz.line(y, x, name=name, win=win,
                          opts=dict(title='LOSS' if name == 'train_loss' else 'Accuracy', xlabel='epoch',
                                    ylabel='loss' if name == 'train_loss' else 'accuracy', showlegend=True))
        else:
            self.viz.line(y, x, name=name, win=win, update='append')
