import os
import numpy as np
import cv2
import torch
from generator import *

class Infer:
    def __init__(self, model_path, output_size=None):
        """
        Initialize the model.
        :param model_path: Path to the trained GFP-GAN model.
        :param output_size: Tuple (width, height) specifying the final output size. If None, original input size is used.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.netG = GFPGANv1Clean(out_size=512,
                                  channel_multiplier=2,
                                  fix_decoder=False,
                                  input_is_latent=True,
                                  different_w=True,
                                  sft_half=True
                                  ).to(self.device)
        self.netG.load_state_dict(torch.load(model_path, map_location=torch.device(self.device))['g_ema'])
        self.netG.eval()
        self.output_size = output_size  # Custom output size defined by user

    def run(self, img_paths, save):
        os.makedirs(save, exist_ok=True)
        for i, img_path in enumerate(img_paths):
            self.run_single(img_path, save)
            print(f'\rProcessing Completed for Image No. {i + 1}', end='', flush=True)
        print()

    def run_single(self, img_path, save):
        """
        Process a single image.
        :param img_path: Path to the input image.
        :param save: Directory where output images will be saved.
        """
        inp, original_shape = self.preprocess(img_path)
        with torch.no_grad():
            oup, _ = self.netG(inp)  # Run inference
        oup = self.postprocess(oup, original_shape)  # Resize back to original or custom size

        img_name = os.path.basename(img_path)
        name, ext = os.path.splitext(img_name)
        output_path = os.path.join(save, f"{name}_output{ext}")
        cv2.imwrite(output_path, oup)

    def preprocess(self, img):
        """
        Resize image to 512x512 before passing into the model.
        :param img: Image file path or numpy array.
        :return: Tensor ready for inference and the original image shape.
        """
        if isinstance(img, str):
            img = cv2.imread(img)
        original_shape = img.shape[:2]  # Store original height and width

        img = cv2.resize(img, (512, 512))  # Resize to 512x512
        img = img.astype(np.float32)[..., ::-1] / 127.5 - 1  # Normalize
        return torch.from_numpy(img.transpose(2, 0, 1)[np.newaxis, ...]).to(self.device), original_shape

    def postprocess(self, img, original_shape):
        """
        Convert model output back to an image and resize it.
        :param img: Model output tensor.
        :param original_shape: Original (height, width) before 512x512 resizing.
        :return: Processed image in the user-defined size.
        """
        img = (torch.clip(img, -1, 1)[0].permute(1, 2, 0).cpu().numpy()[..., ::-1] + 1) * 127.5
        img = img.astype(np.uint8)

        upscaled_shape = (original_shape[1] * 4, original_shape[0] * 4)  # Model enlarges the image by 4×
        img = cv2.resize(img, upscaled_shape)  # Resize to 4x original size

        if self.output_size:  # If custom output size is provided
            img = cv2.resize(img, self.output_size)

        return img

if __name__ == "__main__":
    model_path = r"C:\Users\ACER\Desktop\GFP-GAN Inference Code\gfpgan_final_model.pth"
    base_lq = r'C:\Users\ACER\Desktop\GFP-GAN Inference Code\ffhq-dataset(1024)\biraj-samples_lq'
    
    # Set the output size (e.g., 512x512). Use None to retain original size
    output_size = (503, 723)  # Change this as needed

    model = Infer(model_path, output_size=output_size)

    def get_image_paths(folder):
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    img_paths = get_image_paths(base_lq)
    save = r'C:\Users\ACER\Desktop\GFP-GAN Inference Code\ffhq-dataset(1024)\locally_saved_demos'
    model.run(img_paths, save)
