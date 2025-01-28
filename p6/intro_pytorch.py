import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set=datasets.FashionMNIST("./data",train=True, download=True,transform=custom_transform)
    test_set=datasets.FashionMNIST("./data", train=False, transform=custom_transform)
    if training == True:
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
    else:
        loader = torch.utils.data.DataLoader(test_set, batch_size = 64)
    return loader


def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model
    




def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for epoch in range(T):
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * labels.size(0)
        accuracy = 100 * correct / total
        avg_test_loss = running_loss / total
        print(f"Train Epoch: {epoch} Accuracy: {correct}/{total}({accuracy:.2f}%) Loss: {avg_test_loss:.3f}")

    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    avg_test_loss = test_loss /total
    if show_loss:
        print(f"Average loss: {avg_test_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    


def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    image = test_images[index].unsqueeze(0)
    with torch.no_grad():
        logits = model(image)
    probabilities = F.softmax(logits, dim=1)
    top_probabilities, top_labels = torch.topk(probabilities, 3)
    top_probabilities = top_probabilities.squeeze()
    top_labels = top_labels.squeeze()
    for i in range(3):
        print(f"{class_labels[top_labels[i].item()]}: {top_probabilities[i].item()*100:.2f}%")


# if __name__ == '__main__':
#     '''
#     Feel free to write your own test code here to exaime the correctness of your functions. 
#     Note that this part will not be graded.
#     '''
#     criterion = nn.CrossEntropyLoss()

#     train_loader = get_data_loader()
#     test_loader = get_data_loader(False)
#     model = build_model()
    
#     T = 5
#     train_model(model, train_loader, criterion, T)
#     evaluate_model(model, test_loader, criterion, show_loss = True)
#     test_images = next(iter(test_loader))[0]
#     predict_label(model, test_images, 1)

