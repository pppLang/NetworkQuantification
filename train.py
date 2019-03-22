import torch
from utils import updataConvWei


def train_epoch(model, optimizer, train_loader, criterion, epoch, writer=None, current_lr=None):
    num = len(train_loader)
    model.train()
    for i, (data, label) in enumerate(train_loader):
        model.zero_grad()
        optimizer.zero_grad()
        data = data.cuda()
        label = label.cuda().long()
        # print(data.shape)
        result = model(data)
        loss = criterion(result, label)
        loss.backward()
        # updataConvWei(model, current_lr)
        optimizer.step()
        if i%10==0:
            print('epoch {}, [{}/{}], loss {}'.format(epoch, i, num, loss))

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.cuda()
            label = label.cuda()
            result = model(data)
            test_loss += criterion(result, label)
            pred = result.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    print('test loss {}, acc [{}/{}]'
    .format(test_loss, correct, len(test_loader.dataset)))