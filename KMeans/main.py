from model import KMeans
from utils import get_image, show_image, save_image, error, plot_line


def main():
    # get image
    image = get_image('image.jpg')
    img_shape = image.shape

    # reshape image
    image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    # create model
    num_clusters = [2,5,10,20,50] 
    error_list = []
    for i in num_clusters:
        kmeans = KMeans(i)

        # fit model
        kmeans.fit(image)

        # replace each pixel with its closest cluster center
        image_clustered = kmeans.replace_with_cluster_centers(image)

        # reshape image
        image_clustered = image_clustered.reshape(img_shape)

        # Print the error
        e = error(image, image_clustered)
        error_list.append(e)
        print('MSE:', e )

        # show/save image
        # show_image(image)
        save_image(image_clustered, f'image_clustered_{i}.jpg')
        
    plot_line(num_clusters,error_list,"clusters","Error","Error vs number of clusters")



if __name__ == '__main__':
    main()
