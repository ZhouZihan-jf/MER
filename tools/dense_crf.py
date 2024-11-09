import torch
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch.nn.functional as F
import cv2


def dense_crf(image, output_probs, iterations=1):
    """
    Apply DenseCRF on the network output probabilities.
    
    Args:
    - image (np.array): The original image.
    - output_probs (np.array): The output probabilities from the network (num_classes, H, W).
    - iterations (int): The number of iterations for CRF.
    
    Returns:
    - np.array: The refined output probabilities (H, W).
    """
    
    H, W = image.shape[:2]
    num_classes = output_probs.shape[0]
    
    # Convert the network output to the shape required by DenseCRF
    # output_probs = output_probs.transpose(1, 2, 0)  # (H, W, num_classes)

    unary = utils.unary_from_softmax(output_probs)  # (num_classes, H, W)
    unary = np.ascontiguousarray(unary)
    
    # Initialize DenseCRF
    d = dcrf.DenseCRF2D(W, H, num_classes)
    d.setUnaryEnergy(unary)
    
    # Add pairwise Gaussian and Bilateral potentials
    d.addPairwiseGaussian(sxy=3, compat=1)
    d.addPairwiseBilateral(sxy=50, srgb=15, rgbim=np.ascontiguousarray(image), compat=1)
    
    # Run inference
    Q = d.inference(iterations)
    Q = np.array(Q).reshape((num_classes, H, W))
    
    return Q


def dense_crf_torch(image, output_probs, iterations=3):
    """
    Apply DenseCRF on the network output probabilities.
    
    Args:
    - image (torch.Tensor): The original image (B, C, H, W).
    - output_probs (torch.Tensor): The output probabilities from the network (B, num_classes, H, W).
    - iterations (int): The number of iterations for CRF.
    
    Returns:
    - torch.Tensor: The refined output probabilities (B, num_classes, H, W).
    """
    B, C, H, W = image.shape
    _, num_classes, _, _ = output_probs.shape
    
    refined_outputs = []
    
    for b in range(B):
        # Convert tensors to numpy arrays
        image_np = image[b].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        output_probs_np = output_probs[b].detach().cpu().numpy()
        
        # Apply DenseCRF
        refined_output_np = dense_crf(image_np, output_probs_np, iterations)
        
        # Convert back to torch tensor
        refined_output_torch = torch.from_numpy(refined_output_np).to(output_probs.device)
        refined_outputs.append(refined_output_torch)
    
    # Stack all refined outputs into a tensor
    refined_outputs_tensor = torch.stack(refined_outputs, dim=0)
    
    return refined_outputs_tensor


# Example usage
if __name__ == "__main__":
    # Load your image and PyTorch model output here
    image_path = "/dataset/zzh/DAVIS/JPEGImages/480p/bear/00000.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    
    # Assume `model_output` is the output from your PyTorch model with shape (batch_size, num_classes, H, W)
    # For this example, let's create a dummy output
    batch_size, num_classes, H, W = 1, 7, image.shape[2], image.shape[3]
    model_output = torch.randn((batch_size, num_classes, H, W))
    
    # Apply DenseCRF to the batch of images
    refined_output = dense_crf_torch(image, model_output)
    
    # The `refined_output` now contains the DenseCRF-refined output
    print(refined_output.shape)