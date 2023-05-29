import numpy as np
import matplotlib.pyplot as plt

def training(epochs, model, trainloader, validloader, optimizer, loss_fn):
    loss_training = []
    loss_testing = []
    for e in range(epochs):
        train_loss = 0.0
        model.train()
        for data, labels in trainloader:
            # Clear the gradients
            optimizer.zero_grad()
            # Forward Pass
            target = model(data)
            # Find the Loss
            loss = loss_fn(target,labels)
            # Calculate gradients 
            loss.backward()
            # Update Weights
            optimizer.step()
            # Calculate Loss
            train_loss += loss.item()
        loss_training.append(train_loss / len(trainloader)) 
        valid_loss = 0.0
        model.eval()     
        for data, labels in validloader:
            # Forward Pass
            target = model(data)
            # Find the Loss
            loss = loss_fn(target,labels)
            # Calculate Loss
            valid_loss += loss.item()
        loss_testing.append(valid_loss / len(validloader))
        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}')
    line1, = plt.plot(loss_training, label='training loss')
    line2, = plt.plot(loss_testing, label='validation loss')
    plt.legend(handles=[line1, line2])
    plt.ylim([20, 100])
    plt.show()