from sklearn.metrics import accuracy_score
from timeit import default_timer as timer
import torch
from tqdm.auto import tqdm

def print_train_time(start: float, end: float, device: torch.device = None):
    """
    Helper function to print train time
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds\n")
    return total_time


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
    """
    Train step for a PyTorch model for one epoch
    Turns a PyTorch model to training mode and then
    runs through all of the required training steps (forward pass, 
    loss calculation, optimizer step)

    Args:
        model: A PyTorch model to be trained
        data_loader: A DataLoader instance for the model to be trained on
        loss_fn: A PyTorch loss function to minimize
        optimizer: A PyTorch optimizer to help minimize the loss function
        accuracy_fn: It uses the sklearn accuracy_score function to calculate the accuracy
        device: A target device to compute on (e.g. "cuda" or "cpu")
    
    Returns:
        A dictionary of training loss and training accuracy metrics 
        as well the model name.
    """

    model.train()
    
    train_loss = 0
    train_acc = 0
    # Add a loop to loop through the training batches
    for batch, (X, y) in enumerate(data_loader):
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss

        # Calc accuracy
        #train_acc += accuracy_fn(y.to('cpu'), y_pred.argmax(dim=1).to('cpu'))
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred)

        
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        #if batch % 10 == 0:
        #    print(f'Looked at {batch * len(X)}/{len(data_loader.dataset)} samples.')

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    #print(f'\nTrain loss: {train_loss:.4f} | Train acc: {train_acc:.4f}')
    
    return {'model_name': model.__class__.__name__,
            'train_loss': train_loss.item(),
            'train_acc': train_acc}


def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, 
               accuracy_fn,
               device: torch.device):
    """
    Test step for a PyTorch model for one epoch.
    Turns a PyTorch model to eval mode and then calculates the test 
    loss and test accuracy.
    Args:
        model: A PyTorch model to be trained
        data_loader: A DataLoader instance for the model to be trained on
        loss_fn: A PyTorch loss function to minimize
        accuracy_fn: It uses the sklearn accuracy_score function to calculate the accuracy
        device: A target device to compute on (e.g. "cuda" or "cpu")

    Returns:
        A dictionary of testing loss and testing accuracy metrics
        as well the model name.
        
    """
    
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y)
            test_acc += accuracy_fn(y.to('cpu'), y_pred.argmax(dim=1).to('cpu'))
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)


    #print(f'\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}')

    return {'model_name': model.__class__.__name__,
            'test_loss': test_loss.item(),
            'test_acc': test_acc}

def train(model: torch.nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader,
                 loss_fn: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer,
                 accuracy_fn,
                 epochs: int,
                 device: torch.device):
    """
    Performs the training and evaluation loop for a PyTorch model.
    Args:
        model: A PyTorch model to be trained
        train_dataloader: A DataLoader instance for the model to be trained on
        test_dataloader: A DataLoader instance for the model to be tested on
        loss_fn: A PyTorch loss function to minimize
        optimizer: A PyTorch optimizer to help minimize the loss function
        accuracy_fn: It uses the sklearn accuracy_score function to calculate the accuracy
        epochs: An integer indicating how many epochs to train for
        device: A target device to compute on (e.g. "cuda" or "cpu")
    
    Returns:
        A dictionary of:
            training loss
            training accuracy 
            testing loss
            testing accuracy
            total train time
    """

    model.to(device)
    #print(f'Training model on {device}')
    
    torch.manual_seed(42)

    train_time_start = timer()
    
    epochs = epochs

    results = {
        'train_loss':[],
        'train_acc':[],
        'test_loss':[],
        'test_acc':[],
    }
    
    for epoch in tqdm(range(epochs)):
        #print(f"Epoch: {epoch} ---")
    
        ### Training
        train_score_dict = train_step(model=model,
                                      data_loader=train_dataloader,
                                      loss_fn=loss_fn, 
                                      optimizer=optimizer,
                                      accuracy_fn=accuracy_score,
                                      device=device)

        train_loss = train_score_dict.get('train_loss')
        results.get('train_loss').append(train_loss)
        train_acc = train_score_dict.get('train_acc')
        results.get('train_acc').append(train_acc)
        
        ### Testing
        test_score_dict = test_step(model=model, 
                                    data_loader=test_dataloader, 
                                    loss_fn=loss_fn,
                                    accuracy_fn=accuracy_score,
                                    device=device)

        test_loss = test_score_dict.get('test_loss')
        results.get('test_loss').append(test_loss)
        test_acc = test_score_dict.get('test_acc')
        results.get('test_acc').append(test_acc)

        print(f'Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} |Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}')
            
    
    train_time_end = timer()
    total_train_time = print_train_time(train_time_start, 
                                                train_time_end, 
                                                device=str(next(model.parameters()).device))
    
    results['total_train_time'] = total_train_time
    return results