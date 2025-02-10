import torch


class AverageBasedAutoExposure:
    def __init__(self, Mwhite=255.0):
        """
        Initialize the auto exposure control.

        Args:
            Mwhite (float): The target white level, typically 255 for an 8-bit image.
        """
        self.Mwhite = Mwhite

    def compute_mean(self, image):
        """
        Compute the mean pixel value of the input image.

        Args:
            image (torch.Tensor): The input image tensor (Batch x Channels x Height x Width).

        Returns:
            float: The mean pixel value of the image.
        """
        return torch.mean(image)

    def get_gamma(self, image):
        """
        Calculate the gamma adjustment based on the current image.

        Args:
            image (torch.Tensor): The input image tensor (Batch x Channels x Height x Width).

        Returns:
            float: The computed gamma value to be passed to the camera.
        """
        Imean = self.compute_mean(image)
        gamma = 0.5 * self.Mwhite / Imean
        return gamma

    def adjust_exposure(self, image):
        """
        Adjust the exposure of the image based on the average-based AE scheme.

        Args:
            image (torch.Tensor): The input image tensor (Batch x Channels x Height x Width).

        Returns:
            torch.Tensor: The exposure-adjusted image.
        """
        # Compute the gamma adjustment factor
        gamma = self.get_gamma(image)

        # Adjust the exposure by multiplying the image by the gamma factor
        adjusted_image = image * gamma

        # Clamp the values to avoid going beyond the valid pixel range (0 to Mwhite)
        adjusted_image = torch.clamp(adjusted_image, 0, self.Mwhite)

        return adjusted_image, gamma


# if __name__ == '__main__':
#     # Debugging the AverageBasedAutoExposure class

#     # Create a random image (batch size 1, 3 channels, 128x128 pixels)
#     # Values range between 0 and 255 to simulate an 8-bit image
#     image = torch.rand(1, 3, 128, 128) * 255

#     # Initialize the auto exposure class with a default white level of 255
#     ae = AverageBasedAutoExposure(Mwhite=255.0)

#     # Print original mean pixel value
#     original_mean = ae.compute_mean(image)
#     print(f"Original Mean Pixel Value: {original_mean.item()}")

#     # Apply the auto exposure adjustment and get the new gamma value
#     adjusted_image, gamma = ae.adjust_exposure(image)

#     # Print adjusted mean pixel value and the gamma used
#     adjusted_mean = ae.compute_mean(adjusted_image)
#     print(f"Adjusted Mean Pixel Value: {adjusted_mean.item()}")
#     print(f"Gamma (Exposure Adjustment Factor): {gamma.item()}")
