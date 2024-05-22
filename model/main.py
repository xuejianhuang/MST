import argparse
import logging
import torch
import torch.optim as optim
import time
from transformers import AdamW
from util import *
from data import *
from models import *
from torch_geometric.data import DataLoader
from config import base_Config
import warnings
warnings.filterwarnings('ignore')

#===================train==================
def train(train_dict,model,device,args,vis,epoch,logger):
    train_x_propagation_node_indices, train_x_propagation_node_values, train_x_propagation_idx, \
    train_target, train_root_idx = train_dict['x_p_indices'], train_dict['x_p_values'], train_dict['idx_p'],\
                                   train_dict['y'], train_dict['root_idx_p']
    traindata_set = loadData(train_x_propagation_idx, train_x_propagation_node_indices,
                             train_x_propagation_node_values, train_target, train_root_idx)

    train_loader = DataLoader(traindata_set, batch_size=args.batch, shuffle=True, num_workers=5)
    criterion_clf = nn.BCELoss()
    criterion_clf.to(device)
    optimizer= optim.Adam(model.parameters(),lr=args.lr) #AdamW(model.parameters,lr=args.lr)
    # decayRate = 0.96
    # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    model.train()
    train_loss_list=[]
    total_acc_num=0
    total_num=0
    for idx,Batch_data in enumerate(train_loader):
        Batch_data.to(device)
        optimizer.zero_grad()
        output = model(Batch_data)
        loss = criterion_clf(output, Batch_data.target)
        loss.backward()
        optimizer.step()
       # my_lr_scheduler.step()
        correct = torch.sum(torch.eq(torch.max(output, 1)[1].data, torch.max(Batch_data.target, 1)[1].data)).item()
        num = Batch_data.target.shape[0]
        total_acc_num += correct
        total_num += num
        train_loss_list.append(loss.cpu().item())
        logger.info('Epoch{} | Batch{} | Train_Loss {:.4f} | Train_Accuracy {:.4f}'.format((epoch + 1), \
                                                                                     (idx+1), loss.item(),
                                                                                     correct/num * 100))
    train_loss=np.mean(train_loss_list)
    train_acc=total_acc_num / total_num * 100
    vis.append([train_loss], [epoch], name='train_loss', win='loss', epoch=epoch)
    vis.append([train_acc], [epoch], name='train_acc', win='acc', epoch=epoch)
    return train_acc,train_loss

#===================validation==================
def val(val_dict, model, device,args,vis,epoch):
    val_x_propagation_node_indices, val_x_propagation_node_values, val_x_propagation_idx, \
    val_target, val_root_idx = val_dict['x_p_indices'], val_dict['x_p_values'], val_dict['idx_p'], \
                               val_dict['y'], val_dict['root_idx_p']
    valdata_set = loadData(val_x_propagation_idx, val_x_propagation_node_indices, val_x_propagation_node_values, \
                           val_target, val_root_idx)
    val_loader = DataLoader(valdata_set, batch_size=args.batch, shuffle=True, num_workers=5)
    criterion_clf = nn.BCELoss()
    criterion_clf.to(device)
    model.eval()
    with torch.no_grad():
        val_loss_list = []
        total_acc_num = 0
        total_num = 0
        for Batch_data in val_loader:
            Batch_data.to(device)
            output = model(Batch_data)
            loss = criterion_clf(output, Batch_data.target)
            correct = torch.sum(torch.eq(torch.max(output, 1)[1].data, torch.max(Batch_data.target, 1)[1].data)).item()
            num = Batch_data.target.shape[0]
            total_acc_num += correct
            total_num += num
            val_loss_list.append(loss.cpu().item())

        val_loss = np.mean(val_loss_list)
        val_acc = total_acc_num / total_num * 100
        vis.append([val_loss], [epoch], name='val_loss', win='loss', epoch=epoch)
        vis.append([val_acc], [epoch], name='val_acc', win='acc', epoch=epoch)

    return val_acc,val_loss

#===================test==================
def test(test_dict, model, device):
    model.eval()
    test_result_dict = {}
    with torch.no_grad():
        test_x_propagation_node_indices, test_x_propagation_node_values, test_x_propagation_idx, \
        test_target, test_root_idx = test_dict['x_p_indices'], test_dict['x_p_values'], test_dict['idx_p'], \
                                   test_dict['y'], test_dict['root_idx_p']
        testdata_set = loadData(test_x_propagation_idx, test_x_propagation_node_indices,
                                test_x_propagation_node_values, test_target, test_root_idx)
        test_loader = DataLoader(testdata_set, batch_size=args.batch, shuffle=False, num_workers=5)

        for idx, Batch_data in enumerate(test_loader):
            Batch_data.to(device)
            output = model(Batch_data)
            output = torch.max(output, 1)[1].data
            target_new = torch.max(Batch_data.target, 1)[1].data
            if idx == 0:
                output_all = output
                label_all = target_new
            else:
                output_all = torch.cat([output_all,output],dim=0)
                label_all = torch.cat([label_all,target_new],dim=0)
        acc = accuracy(output_all, label_all)
        f1, precision, recall, f1_real, precision_real, recall_real, f1_fake, precision_fake, recall_fake = \
            macro_f1(output_all, label_all, num_classes=2)
    test_result_dict['acc'] = acc
    test_result_dict['prec'] = precision
    test_result_dict['rec'] = recall
    test_result_dict['f1'] = f1

    test_result_dict['prec_fake'] = precision_fake
    test_result_dict['rec_fake'] = recall_fake
    test_result_dict['f1_fake'] = f1_fake

    test_result_dict['prec_real'] = precision_real
    test_result_dict['rec_real'] = recall_real
    test_result_dict['f1_real'] = f1_real

    return test_result_dict

def parse_arguments():
    parser = argparse.ArgumentParser(description='dualDynamicGCN')
    parser.add_argument('--dataset', type=str, default='weibo')
    parser.add_argument('--model', type=str, default='user')  # propagation/semantic/user/dual/wotf/static
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=0, help="The random seed for initialization.")
    args = parser.parse_args()
    return args


#====================main=======================
if __name__ == "__main__":
    args=parse_arguments()
    logger=set_logger(args)
    #set_torch_seed(args.seed)
    config=base_Config()
    log_args(logger,args)
    log_config(logger,config)
    vis = Visualize(server='localhost', host=8097)
    model,train_dict,val_dict,test_dict=getModelAndData(config,args.model,args.dataset)

    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('total number of parameters:{}'.format(pytorch_total_trainable_params))
    start_time = time.time()
    model = model.to(config.device)
    best_acc=0
    early_stop_cnt=0
    model_saved_path=config.model_saved_path+args.dataset+"_"+args.model+".pkl"
    for epoch in range(args.epoch):
        logger.info('-----------Epoch:'+str(epoch)+"-----------")
        train_acc, train_loss = train(train_dict,model,config.device,args,vis,epoch,logger)
        logger.info('train_loss:{:.5f} train_acc:{:.3f}'.format(train_loss, train_acc))
        val_acc,val_loss=val(val_dict,model,config.device,args,vis,epoch)
        logger.info("val_acc:{:.5f} val_acc:{:.3f} \n".format(val_loss, val_acc))
        if val_acc>best_acc:
            best_acc=val_acc
            torch.save(model.state_dict(), model_saved_path)
            logger.info("save model,acc:{:.3f}".format(best_acc))
            early_stop_cnt=0
        else:
            early_stop_cnt+=1
        if early_stop_cnt >= config.patience:
            break
    model.load_state_dict(torch.load(model_saved_path))
    test_result_dict=test(test_dict,model,config.device)
    saveTestResult(args.dataset,args.model,test_result_dict)
    log_testResult(logger,test_result_dict)
    end_time = time.time()
    logger.info("the running time is: {:.1f} s".format(end_time - start_time))
