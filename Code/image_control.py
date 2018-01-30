from PIL import Image
import numpy as np



class IM_CONTROL:
    def __init__(self, path):
        self.image = Image.open(path).convert("LA")
        self.image.save("greyscale.png")

    def grey_scale_matrix(self):
        """
        Return matrix where elements are grey scale value (0-255) of pixel 
        at row, column.
        """
        width, height = self.image.size[0], self.image.size[1]
        array = np.asarray(self.image)
        matrix = np.empty((height, width), dtype=np.uint8)
        for row in range(height):
            for column in range(width):
                matrix[row][column] = array[row][column][0]
        return matrix

    def show_image(self, reconstruction):
        """
        Reconstruct image in reduced dimensions. X = ZY.T where Z is reduction
        """
        image = Image.fromarray(reconstruction.astype(np.uint8))
        image.save("../test.png")
