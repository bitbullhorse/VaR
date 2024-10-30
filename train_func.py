import os
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader as Dataloader
# from torch.utils.data.dataloader import Dataloader
import torch

device = 'cuda'
cwd = '/home/czj/pycharm_project_tmp_pytorch/VaR/'
torch.autograd.set_detect_anomaly(True)

def train_model_cp(model, epochs, train_loader:Dataloader, Loss, optimizer, scheduler, eval_loader:Dataloader):
    model.to(device)
    model.train()
    print(len(train_loader.dataset))
    print(len(eval_loader.dataset))
    count = 0
    loss_list = []
    eval_close_list = []
    close_list = []
    for epoch in range(epochs):
        print(f'****************epoch:{epoch}****************')
        for src, tgt, cp in train_loader:
            optimizer.zero_grad()
            count+=1
            output = model(src, tgt)
            loss = Loss(output, cp)
            if count % 100 == 0:
                print("**********************************************")
                print("___________________________________")
                print(cp.view(-1))
                print("___________________________________")
                print(output.view(-1))
                print("___________________________________")
                print("**********************************************")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        loss_sum = 0
        for src, tgt, cp in eval_loader:
            output = model(src, src, has_mask=False)
            loss = Loss(output, cp)
            loss_sum+=loss.item()
        loss_list.append(loss_sum)

    day = 0
    for src, _, cp in eval_loader:
        day+=eval_loader.batch_size
        output = model(src, src, has_mask=False)
        close_list.append(cp.view(-1).tolist())
        eval_close_list.append(output.view(-1).tolist())
        # print(tgt[:, :, 3].view(-1).tolist())

    fig, axes = plt.subplots(2, 1)
    x_label = list(range(len(loss_list)))
    axes[0].plot(loss_list)
    axes[0].set_title('Loss function',)
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    
    x1_label = list(range(len(eval_close_list)))
    axes[1].plot(eval_close_list, 'b-', label=u'predicted price')
    axes[1].plot(close_list, 'r-', label=u'close price')
    axes[1].set_title('Close Price')
    axes[1].set_xlabel('day')
    axes[1].set_ylabel('price')
    plt.legend(loc='upper left', fontsize='large')
    plt.savefig('original_Transformer_loss&close.png')
    plt.show()


def train_iTranformer(model, epochs, train_loader:Dataloader, Loss, optimizer, scheduler, eval_loader:Dataloader):
    model.train()
    model.to(device)
    count = 0
    loss_list = []
    for epoch in range(epochs):
        print(f'****************epoch:{epoch}****************')
        for x, y in train_loader:
            optimizer.zero_grad()
            count += 1
            output = model(x)
            loss = Loss(output, y)
            if count % 100 == 0:
                print("**********************************************")
                print("___________________________________")
                print(y[0].view(-1))
                print("___________________________________")
                print(output[0].view(-1))
                print("___________________________________")
                print("**********************************************")
            loss.backward()
            optimizer.step()

        scheduler.step()
        loss_sum = 0
        for x, y in eval_loader:
            output = model(x)
            loss = Loss(y, output)
            loss_sum += loss.item()
        loss_list.append(loss_sum)

    label = []
    pred = []
    for x, y in eval_loader:
        output = model(x)[:, :, 0].view(-1).tolist()
        y = y[:, :, 0].view(-1).tolist()
        label+=y
        pred+=output
    print(label)
    print(pred)
    plt.figure(figsize=(9, 12))
    plt.plot(label, 'b-', label='label')
    plt.plot(pred, 'r-', label='predict')
    plt.legend()
    plt.show()
