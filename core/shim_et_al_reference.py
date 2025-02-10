import torch
import torch.nn.functional as F
import numpy as np

class GradientExposureControl:
    def __init__(self, scale=1.0, Lambda=10, delta=0.01, n_points=61, default_K_p=0.2, gamma_anchors=None):
        self.scale = scale
        self.Lambda = Lambda
        self.delta = delta
        self.n_points = n_points
        self.K_p = default_K_p
        self.gamma_anchors = gamma_anchors if gamma_anchors is not None else torch.linspace(0.1, 1.9, steps=n_points)

    def gradient_exposure_control(self, im_cap, grad_update='linear'):
        # Compute gradient information
        grad_info_list = self.make_gradient_info(im_cap, self.gamma_anchors, self.scale, self.Lambda, self.delta)

        # Interpolate with cubic splines
        gamma_interp = torch.linspace(0.1, 1.9, steps=self.n_points)
        gi_interp = self.make_spline(self.gamma_anchors, grad_info_list, gamma_interp)

        # Compute gamma that maximizes the gradient information (use torch's argmax)
        i_max = torch.argmax(gi_interp)
        gamma_hat = gamma_interp[i_max]

        # Compute the next exposure
        new_exp = self.exposure_update(gamma_hat, grad_update=grad_update, K_p=self.K_p)
        return new_exp

    def exposure_update(self, gamma_hat, current_exp=2, grad_update='linear', K_p=0.4):
        d_NL_update = 0.5

        if grad_update == 'linear':
            r = 1 - gamma_hat
        else:
            r = d_NL_update * torch.tan((1 - gamma_hat) * torch.atan(1 / d_NL_update))

        r = r.float()

        # Calculate new exposure based on gamma_hat
        new_exp = torch.where(gamma_hat >= 1,
                              (1 + 0.5 * K_p * r) * current_exp,
                              (1 + K_p * r) * current_exp)
        return new_exp

    def make_gradient_info(self, img_input, gamma_anchors, scale, Lambda, delta):

        img_tf = img_input
        img_tf.view(img_tf.size(0), img_tf.size(1), img_tf.size(2) // 2, 2, img_tf.size(3) // 4, 4).sum(dim=(3, 5))

        img_tf = img_tf.mean(dim=1, keepdim=True)

        sobol_horizontal = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).unsqueeze(0).unsqueeze(0)
        sobol_vertical = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).unsqueeze(0).unsqueeze(0)

        device = img_input.device  # cuda
        sobol_horizontal = sobol_horizontal.to(device)
        sobol_vertical = sobol_vertical.to(device)

        N = torch.log(torch.tensor(Lambda * (1 - delta) + 1))

        grad_info_list = []

        for gamma in gamma_anchors:
            # Apply gamma correction
            img = (torch.pow(img_tf * scale, gamma)) * 0.25
            img_pad = F.pad(img, (3, 3, 3, 3), mode='reflect')

            # Apply Sobel filters
            img_grad_x = F.conv2d(img_pad, sobol_horizontal)
            img_grad_y = F.conv2d(img_pad, sobol_vertical)

            grad_norm = torch.sqrt(img_grad_x ** 2 + img_grad_y ** 2)[:, :, 2:-2, 2:-2]
            grad_info = torch.log(torch.tensor(Lambda) * F.relu(grad_norm - delta) + 1) / N
            grad_info = grad_info.sum()

            grad_info_list.append(grad_info)

        return torch.stack(grad_info_list)

    def make_spline(self, x, y, qx):
        """ 1D cubic spline interpolation in PyTorch """
        device = x.device  
        x = x.to(device)
        y = y.to(device)
        qx = qx.to(device)

        nx = len(x)
        h = x[1:] - x[:-1]
        A_diag = 2 * (h[1:] + h[:-1])
        zero_cell = torch.tensor([0], dtype=x.dtype, device=device)
        one_cell = torch.tensor([1], dtype=x.dtype, device=device)
        A_diag = torch.cat([one_cell, A_diag, one_cell], dim=0)
        A_upper = torch.cat([zero_cell, h[1:]], dim=0)
        A_lower = torch.cat([h[:-1], zero_cell], dim=0)
        A = self.make_tri_diag(A_diag, A_upper, A_lower)

        B = 3.0 * (y[2:] - y[1:-1]) / h[1:] - 3.0 * (y[1:-1] - y[:-2]) / h[:-1]
        B = torch.cat([zero_cell, B, zero_cell], dim=0).unsqueeze(1)

        # Solving the tridiagonal system
        c_full = torch.linalg.solve(A, B)

        c = c_full.squeeze(1)
        d = (c[1:] - c[:-1]) / (3.0 * h)
        b = (y[1:] - y[:-1]) / h - h * (c[1:] + 2.0 * c[:-1]) / 3.0

        ind = self.search_index(x, qx)
        ind = torch.minimum(ind, torch.tensor(nx - 2, device=device))

        x_i = x[ind]
        dx = qx - x_i
        y_i = y[ind]
        b_i = b[ind]
        c_i = c[ind]
        d_i = d[ind]

        result = y_i + b_i * dx + c_i * dx ** 2 + d_i * dx ** 3
        return result

    def make_tri_diag(self, diag, upper, lower):
        """ Helper function for tridiagonal system """
        diag = diag.unsqueeze(1)
        lower = lower.unsqueeze(1)
        upper = upper.unsqueeze(1)

        mat = torch.zeros(diag.size(0), diag.size(0), dtype=diag.dtype)
        for i in range(0, diag.size(0)-1):
            mat[i, i - 1] = lower[i]
            mat[i - 1, i] = upper[i - 1]

        mat += torch.diag(diag.squeeze())
        return mat

    def search_index(self, x, qx):
        """ Helper function for searching index """
        x_grid = x.unsqueeze(1).repeat(1, len(qx))
        x_match = (qx >= x_grid).int()
        index = x_match.sum(dim=0) - 1
        return index

# if __name__ == "__main__":
#     # Create a dummy image tensor with shape (batch_size, channels, height, width)
#     batch_size, channels, height, width = 1, 3, 640, 480
#     im_cap = torch.randn(batch_size, channels, height, width)

#     # Instantiate the class
#     control = GradientExposureControl(scale=1.0, Lambda=10, delta=0.01, n_points=61, default_K_p=0.2)

#     # Run the exposure control algorithm
#     new_exposure = control.gradient_exposure_control(im_cap, grad_update='linear')

#     print(f"New Exposure: {new_exposure}")
