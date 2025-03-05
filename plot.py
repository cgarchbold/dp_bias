import matplotlib.pyplot as plt


def plot_losses(history, save_pth):
    epochs_list, train_losses, val_losses = zip(*history)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_list, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs_list, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid()
    plt.savefig(save_pth)