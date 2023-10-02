from matplotlib import pyplot as plt
import numpy as np

def get_image(image_path):
    image = plt.imread(image_path)
    return image/255.0

def show_image(image):
    plt.imshow(image)
    plt.show()

def save_image(image, image_path):
    plt.imsave(image_path, image)


def error(original_image: np.ndarray, clustered_image: np.ndarray) -> float:
    # Returns the Mean Squared Error between the original image and the clustered image
    original_image=original_image.reshape(clustered_image.shape)
    return np.mean(np.square(original_image-clustered_image))


def plot_line(X: list, Y : list, X_label : str, Y_label :str , title):
    fig,ax = plt.subplots()
    ax.set_xlabel(X_label)
    ax.set_ylabel(Y_label)
    ax.set_title(title)
    ax.plot(X, Y,"--o" ,color='blue')
    
    plt.savefig('Error_W.r.t_clusters.jpg')