import torch
import torch.nn.functional as F

def train(model, device, x_train, t_train, optimizer, epoch):
    model.train()
    
    for start in range(0, len(t_train)-1, 256):
      end = start + 256
      x, y = torch.from_numpy(x_train[start:end]), torch.from_numpy(t_train[start:end]).long()
      x, y = x.to(device), y.to(device)
      
      optimizer.zero_grad()

      output = model(x)
      loss = F.cross_entropy(output, y)
      loss.backward()
      optimizer.step()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))

def test(model, device, x_test, t_test):
    model.eval()
    test_loss = 0
    correct = 0
    for start in range(0, len(t_test)-1, 256):
      end = start + 256
      with torch.no_grad():
        x, y = torch.from_numpy(x_test[start:end]), torch.from_numpy(t_test[start:end]).long()
        x, y = x.to(device), y.to(device)
        output = model(x)
        test_loss += F.cross_entropy(output, y).item() # sum up batch loss
        pred = output.max(1, keepdim=True)[1] # get the index of the max logit
        correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= len(t_test)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(t_test),
        100. * correct / len(t_test)))
    return 100. * correct / len(t_test)

def on_task_update(model, device, optimizer, fisher_dict, optpar_dict, task_id, x_mem, t_mem):
    
    model.train()
    optimizer.zero_grad()

    # accumulating gradients
    for start in range(0, len(t_mem)-1, 256):
        end = start + 256

        x, y = torch.from_numpy(x_mem[start:end]), torch.from_numpy(t_mem[start:end]).long()
        x, y = x.to(device), y.to(device)

        output = model(x)

        loss = F.cross_entropy(output, y)
        loss.backward()

    # gradients accumulated can be used to calculate fisher
    fisher_dict[task_id] = {}
    optpar_dict[task_id] = {}

    for name, param in model.named_parameters():
        optpar_dict[task_id][name] = param.data.clone()
        fisher_dict[task_id][name] = param.grad.data.clone().pow(2)

    return fisher_dict, optpar_dict

def train_ewc(model, device, fisher_dict, optpar_dict, ewc_lambda, task_id, x_train, t_train, optimizer, epoch):
    
    model.train()

    for start in range(0, len(t_train)-1, 256):

        end = start + 256
        x, y = torch.from_numpy(x_train[start:end]), torch.from_numpy(t_train[start:end]).long()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(x)
        loss = F.cross_entropy(output, y)

        ### magic here!
        for task in range(task_id):
            for name, param in model.named_parameters():
                fisher = fisher_dict[task][name]
                optpar = optpar_dict[task][name]
                loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda

        loss.backward()
        optimizer.step()

    print('Train Epoch : {} \t Loss : {:6f}'.format(epoch, loss.item()))

    return fisher_dict, optpar_dict