import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net
from mtevi import EvidentialnetMarginalLikelihood, EvidenceRegularizer, modified_mse


nll_loss = EvidentialnetMarginalLikelihood() ##  NLL loss
reg = EvidenceRegularizer() ## evidential regularizer
mmse_loss = modified_mse ## lipschitz MSE loss

loss_function = nn.NLLLoss()
# can be changed according to config.txt
exer_n = 17746
knowledge_n = 123
student_n = 4163
# can be changed according to command parameter
device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
epoch_n = 5
batch_size = 32
aa,bb,cc,dd,ee = 1,1,1,1,1
def train():
    data_loader = TrainDataLoader()
    net = Net(student_n, exer_n, knowledge_n)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    print('training model...')



    for epoch in range(epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()

            gamma, nu, alpha, beta = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)

            output_0 = torch.ones(gamma.size()).to(device) - gamma
            output = torch.cat((output_0, gamma), 1)

            # grad_penalty = 0
            lossBM = loss_function(torch.log(output), labels)
            nll__loss = nll_loss(gamma, nu, alpha, beta, labels).mean()
            reg__loss = reg(gamma, nu, alpha, beta, labels).mean()
            mmse__loss = mmse_loss(gamma, nu, alpha, beta, labels).mean()
            loss = (aa*nll__loss + bb*reg__loss + cc*mmse__loss)*dd + ee*lossBM
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0

        # validate and save current model every epoch
        rmse, auc = validate(net, epoch)
        save_snapshot(net, '../model/model_epoch' + str(epoch + 1))


def validate(model, epoch):
    data_loader = ValTestDataLoader('validation')
    net = Net(student_n, exer_n, knowledge_n)
    print('validating model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        output,_,_,_ = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)

    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)


    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))
    with open('../result/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch+1, accuracy, rmse, auc))

    return rmse, auc


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


if __name__ == '__main__':
    if (len(sys.argv) != 3) or ((sys.argv[1] != 'cpu') and ('cuda:' not in sys.argv[1])) or (not sys.argv[2].isdigit()):
        print('command:\n\tpython train.py {device} {epoch}\nexample:\n\tpython train.py cuda:0 70')
        exit(1)
    else:
        device = torch.device(sys.argv[1])
        epoch_n = int(sys.argv[2])

    # global student_n, exer_n, knowledge_n, device

    student_n, exer_n, knowledge_n = 4163, 17746, 123

    train()
