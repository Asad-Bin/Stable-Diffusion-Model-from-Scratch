import matplotlib.pyplot as plt

def local(outputs = None):
    if outputs is None:
        print("No outputs provided for plotting.")
        return
    
    # inside your training loop after outputs computed:
    outputs_cpu = outputs.detach().cpu()  # move to CPU and detach from graph
    
    # outputs shape: [batch_size, 3, 512, 512]
    # pick first image in batch
    img = outputs_cpu[0]

    # convert from [C,H,W] to [H,W,C] for matplotlib
    img = img.permute(1, 2, 0).numpy()

    plt.imshow(img)
    plt.axis('off')
    plt.show()

