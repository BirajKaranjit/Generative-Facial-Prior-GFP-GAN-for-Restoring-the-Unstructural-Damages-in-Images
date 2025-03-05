#GFPGAN Trainer.py
import torch.distributed as dist
from itertools import chain
from torchvision.ops import roi_align
import pdb
from tqdm import tqdm
# from torchvision.models import vgg19, vgg16
from model_trainer import *
from generator import *
from stylegan_arch import *
from discriminator import *
from utils import *
from loss import *
# from model.generator import GFPGANv1Clean
# from model.discriminator import StyleGAN2Discriminator,FacialComponentDiscriminator
# from utils.utils import *
# from model.loss import *
# from model.third.arcface_arch import ResNetArcFace

class GFPTrainer(ModelTrainer):

    def __init__(self, args):
        super().__init__(args)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.netG = GFPGANv1Clean(out_size=args.out_size,
                                channel_multiplier=args.channel_multiplier,
                                fix_decoder=args.fix_decoder,
                                input_is_latent=args.input_is_latent,
                                different_w=args.different_w,
                                sft_half=args.sft_half
                                ).to(self.device)

        self.netD = StyleGAN2Discriminator(1024).to(self.device)
        self.left_eye_d = self.right_eye_d = self.mouth_d = None

        if self.args.crop_components:
            self.left_eye_d = FacialComponentDiscriminator().to(self.device)
            self.right_eye_d = FacialComponentDiscriminator().to(self.device)
            self.mouth_d = FacialComponentDiscriminator().to(self.device)
            init_weights(self.left_eye_d,'xavier_uniform')
            init_weights(self.right_eye_d,'xavier_uniform')
            init_weights(self.mouth_d,'xavier_uniform')
            self.PartGANLoss = GANLoss(gan_type=args.part_gan_type,
                               loss_weight=args.lambda_gan_part)



        self.g_ema = GFPGANv1Clean(out_size=args.out_size,
                                channel_multiplier=args.channel_multiplier,
                                fix_decoder=args.fix_decoder,
                                input_is_latent=args.input_is_latent,
                                different_w=args.different_w,
                                sft_half=args.sft_half
                                ).to(self.device)
        self.g_ema.eval()

        init_weights(self.netD,'xavier_uniform')

        init_weights(self.netG,'xavier_uniform')

        self.optimG,self.optimD = self.create_optimizer()  #This line creates two optimizers: optimG for the generator (netG) and optimD for the discriminator (netD).

        if self.args.scratch:
            self.load_scratch()

        if args.pretrain_path is not None:
            self.loadParameters(args.pretrain_path)

        accumulate(self.g_ema,self.netG,0,args.use_buffer)

        self.netG,self.netG_module = self.use_ddp(self.netG,args.dist,True)   #This line uses self.use_ddp() to wrap the generator model (netG) with Distributed Data Parallel (DDP), which is useful for parallel training across multiple GPUs.
        self.netD,self.netD_module = self.use_ddp(self.netD,args.dist)
        self.left_eye_d,self.left_eye_d_module = self.use_ddp(self.left_eye_d,args.dist)
        self.right_eye_d,self.right_eye_d_module = self.use_ddp(self.right_eye_d,args.dist)
        self.mouth_d,self.mouth_d_module = self.use_ddp(self.mouth_d,args.dist)




        self.L1Loss = nn.L1Loss()

        if self.args.perloss:
            self.PerLoss = PerceptualLoss(args.layer_weights,args.vgg_type,
                                        args.use_input_norm,
                                        args.range_norm,
                                        perceptual_weight=args.lambda_perceptual,
                                        style_weight=args.lambda_style,
                                        criterion=args.criterion).to(self.device)
            requires_grad(self.PerLoss,False)
            self.PerLoss.eval()

        if self.args.idloss:
            self.idModel = ResNetArcFace(self.args.out_size, self.args.id_block,
                                         self.args.id_layers,
                                         self.args.id_use_se).to(self.device)
            # self.idModel.load_state_dict(torch.load(self.args.id_model))
            load_state_dict_with_prefix_removal(self.idModel, self.args.id_model, self.device)
            requires_grad(self.idModel,False)
            self.idModel.eval()

        self.GANLoss = GANLoss(gan_type=args.gan_type,
                               loss_weight=args.lambda_gan)

        self.accum = 0.5 ** (32 / (10 * 1000))

    def create_optimizer(self):

        if self.args.mode == "decoder":
            g_params = self.netG.stylegan_decoder.parameters()
        elif self.args.mode == "encoder":
            requires_grad(self.netG.stylegan_decoder,False)
            g_params = [f for f in self.netG.parameters() if f.requires_grad]
        else:
            g_params = self.netG.parameters()

        g_optim = torch.optim.Adam(
                    g_params,
                    lr=self.args.g_lr,
                    betas=(self.args.beta1, self.args.beta2),
                    )

        if self.args.crop_components:
            d_params = chain(self.left_eye_d.parameters(),
                             self.right_eye_d.parameters(),
                             self.mouth_d.parameters(),
                             self.netD.parameters())
        else:
            d_params = self.netD.parameters()
        d_optim = torch.optim.Adam(
                    d_params,
                    lr=self.args.d_lr,
                    betas=(self.args.beta1, self.args.beta2),
                    )

        return  g_optim,d_optim


    def run_single_step(self, data, steps):

        self.netG.train()
        lq,gt = data  #this is the only line added here, keep or delete it ?? let's see later
        # self.run_discriminator_one_step(data,steps)
        super().run_single_step(data,steps)   #calls run_single_step function from the parent class of GFPGANv1 class i.e. nn.Module


    def run_generator_one_step(self, data,step):


        requires_grad(self.netG, True)

        requires_grad(self.netD, False)
        requires_grad(self.left_eye_d, False)
        requires_grad(self.right_eye_d, False)
        requires_grad(self.mouth_d, False)


        # lq,gt,loc_left_eye, loc_right_eye, loc_mouth = data
        lq, gt = data

        G_losses,loss,fake = \
        self.compute_g_loss(lq,gt,step)
        # self.compute_g_loss(lq,gt,loc_left_eye, loc_right_eye, loc_mouth,step)
        # pdb.set_trace()
        self.optimG.zero_grad()
        loss.mean().backward()
        self.optimG.step()

        accumulate(self.g_ema,self.netG_module,self.accum,self.args.use_buffer)

        self.g_losses = G_losses
        # Store the losses for retrieval
        self.latest_losses = G_losses  #  Store here!
        # print(f"Stored latest_losses: {self.latest_losses}")  # Debugging print

        self.generator = [lq.detach(),fake.detach(),gt.detach()]  #The .detach() method is used to prevent these tensors (lq, fake, gt) from being part of the computational graph. The detach() function is crucial because it ensures that no gradients are computed for these tensors in the current forward pass.





    def evalution(self, test_loader, steps, epoch):
        self.netG.eval()
        self.netD.eval()

        loss_dict = {}
        index = random.randint(0, len(test_loader) - 1)  # Random index for visualization
        counter = 0

        # Wrap the test_loader with tqdm for progress visualization
        val_progress = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{self.args.max_epoch} [Validation]", unit="batch")
        with torch.no_grad():
            for i, data in enumerate(val_progress):
                data = self.process_input(data)
                lq, gt = data  # Unpack data

                # Compute generator and discriminator losses
                G_losses, loss, fake = self.compute_g_loss(lq, gt, steps)
                # loss, D_losses = self.compute_d_loss(fake, gt, steps)
                # G_losses = {**G_losses, **D_losses}  # Combine generator and discriminator losses

                # Accumulate losses
                for k, v in G_losses.items():
                    loss_dict[k] = loss_dict.get(k, 0) + v.detach()

                # Update progress bar with current loss
                val_progress.set_postfix({"Val Loss": loss.item()})  # Display validation loss inline
                # Visualize results for a random batch
                if i == index and self.args.rank == 0:
                    ema_oup, _ = self.g_ema(lq, return_rgb=False)
                    show_data = [lq.detach(), fake.detach(), ema_oup.detach(), gt.detach()]
                    self.val_vis.display_current_results(self.select_img(show_data, size=show_data[-1].shape[-1]), steps)

                    # Save images to disk
                    save_path = os.path.join(self.args.checkpoint_path, "val_images", f"epoch_{epoch}_step_{steps}.png")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    # self.save_images(show_data, save_path)  # Implement a function to save images

                counter += 1

        # Average losses over all batches
        for key, val in loss_dict.items():
            loss_dict[key] /= counter

        # Distributed training: Aggregate losses across all processes
        if self.args.dist:
            dist_losses = loss_dict.copy()
            for key, val in loss_dict.items():
                dist.reduce(dist_losses[key], 0)
                value = dist_losses[key].item()
                loss_dict[key] = value / self.args.world_size

        # Log validation losses to TensorBoard and print to console
        if self.args.rank == 0:
            self.val_vis.plot_current_errors(loss_dict, steps)
            self.val_vis.print_current_errors(epoch, steps, loss_dict, 0)

        # Return validation loss dictionary for early stopping or other purposes
        return loss_dict


    # def compute_g_loss(self,lq,gt,loc_left_eye, loc_right_eye, loc_mouth,step): #because we don't use facial landmarks for eyes and mouth
    def compute_g_loss(self,lq,gt,step):

            G_losses = {}
            loss = 0
            fake, out_rgbs = self.netG(lq, return_rgb=False)
            self.latest_fake_images = fake  #for getting the latest upscaled image for displaying in-line in the terminal march 2

            if self.args.pixloss:
                pixloss = self.L1Loss(fake,gt) * self.args.lambda_l1
                loss += pixloss
                G_losses['g_pixloss'] = pixloss



            if self.args.perloss:

                l_g_percep, l_g_style = self.PerLoss(fake, gt)
                if l_g_percep is not None:
                    loss += l_g_percep
                    G_losses['g_percep_loss'] = l_g_percep
                if l_g_style is not None:
                    loss += l_g_style
                    G_losses['g_style_loss'] = l_g_style

            if self.args.idloss:
                out_gray = self.gray_resize_for_identity(fake)
                gt_gray = self.gray_resize_for_identity(gt)

                identity_gt = self.idModel(gt_gray).detach()
                identity_out = self.idModel(out_gray)
                l_identity = self.L1Loss(identity_out, identity_gt) * self.args.lambda_id
                loss += l_identity
                G_losses['g_id_loss'] = l_identity

            # gan loss
            fake_g_pred,fake_feat = self.netD(fake)
            l_g_gan = self.GANLoss(fake_g_pred, True, is_disc=False)
            loss += l_g_gan
            G_losses['g_gan'] = l_g_gan

            if self.args.featloss:
                _,real_feat = self.netD(gt)
                fm_loss = 0
                for r_f,f_f in zip(real_feat,fake_feat):
                    fm_loss += self.L1Loss(r_f,f_f)

                fm_loss = fm_loss / len(real_feat) * self.args.lambda_fm
                loss += fm_loss
                G_losses['g_fm_loss'] = fm_loss

            G_losses['loss'] = loss
            # Store losses in self.latest_losses
            self.latest_losses = G_losses
            # print(f"\nG_losses at step {step}: {G_losses}")   #Debug point


            return G_losses,loss,fake


    def get_latest_losses(self):
        if not hasattr(self, "latest_losses") or not self.latest_losses:
            print("Warning: get_latest_losses called before losses were stored.")
            return {"loss": 0.0}  # Return a default value instead of an empty dict

        # print(f"Retrieving latest_losses: {self.latest_losses}")  # Debugging print
        return self.latest_losses



    def get_latest_generated(self):
        if hasattr(self, "latest_fake_images"):  # Check if images exist
            return self.latest_fake_images
        else:
            print("Error: No generated images found!")
            return None


    def get_loss_from_val(self,loss):
        total_loss = 0.0
        # Check and add each loss term if it exists
        if 'g_comp_style_loss' in loss:
            total_loss += loss['g_comp_style_loss']
        else:
            print("Warning: 'g_comp_style_loss' not found in loss dictionary. Skipping.")

        if 'g_id_loss' in loss:
            total_loss += loss['g_id_loss']
        else:
            print("Warning: 'g_id_loss' not found in loss dictionary. Skipping.")

        if 'g_percep_loss' in loss:
            total_loss += loss['g_percep_loss']
        else:
            print("Warning: 'g_percep_loss' not found in loss dictionary. Skipping.")

        if 'g_pixloss' in loss:
            total_loss += loss['g_pixloss']
        else:
            print("Warning: 'g_pixloss' not found in loss dictionary. Skipping.")

        if 'g_fm_loss' in loss:
            total_loss += loss['g_fm_loss']
        else:
            print("Warning: 'g_fm_loss' not found in loss dictionary. Skipping.")

        return total_loss


    def loadParameters(self,path):
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)

        self.netG.load_state_dict(ckpt['G'],strict=True)  #he strict=True argument ensures that the model's current architecture exactly matches the structure of the saved model in the checkpoint.
#It means, if there are missing or extra keys in ckpt['G'] or ckpt['D'] compared to self.netG and self.netD, an error will be raised.
        self.netD.load_state_dict(ckpt['D'],strict=True)

        if self.args.crop_components:
            self.left_eye_d.load_state_dict(ckpt['left_eye_d'])
            self.right_eye_d.load_state_dict(ckpt['right_eye_d'])
            self.mouth_d.load_state_dict(ckpt['mouth_d'])

        try:
          self.optimG.load_state_dict(ckpt['g_optim'])
          self.optimD.load_state_dict(ckpt['d_optim'])
          print("Pre-Trained Optimizers are loaded properly.")
        # if 'g_optim' in ckpt and 'd_optim' in ckpt:
        #   self.optimG.load_state_dict(ckpt['g_optim'])
        #   self.optimD.load_state_dict(ckpt['d_optim'])
        except:
          print("Optimizer states not found in checkpoint. Using newly created optimizers.")
          print("It means the state_dict of pre-trained optimizers 'g_optim' and 'd_optim' has not been loaded and the Optimizers will start to train from scratch. It might cause spike in losses in initial epochs")  #this has been displaying when i am starting to train
#it means that the optimizers state_dict 'g_optim' and 'd_optim' are not loaded properly thus, the optimizer states are not restored, meaning training will start from scratch for the optimizers.
#Issues if optimizers previous state_dict is not loaded:-> Since, the model 'final.pth' was well trained i.e. the training was near convergence, and the optim_state_dict are not loaded so, we loose the Adaptive Momentum learned by the optimizers in its previous training. Hence, it will take longer to recover, and loss will spike initially.


    def load_scratch(self):

        state_dict = torch.load(self.args.scratch_gan_path)['params_ema']
        model_dict = self.netG.state_dict()
        state_dict = {k:v for k,v in state_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.netG.load_state_dict(model_dict)

        self.netD.load_state_dict(torch.load(self.args.scratch_d_path))

        if self.args.crop_components:
            self.left_eye_d.load_state_dict(torch.load(self.args.scratch_left_eye_path)['params'])
            self.right_eye_d.load_state_dict(torch.load(self.args.scratch_right_eye_path)['params'])
            self.mouth_d.load_state_dict(torch.load(self.args.scratch_mouth_path)['params'])



    def saveParameters(self,path):

        save_dict = {
            "G": self.netG_module.state_dict(),
            'g_ema':self.g_ema.state_dict(),
            "g_optim": self.optimG.state_dict(),
            'D':self.netD_module.state_dict(),
            'd_optim': self.optimD.state_dict(),
            "args": self.args
        }

        if self.args.crop_components:
            save_dict['left_eye_d'] = self.left_eye_d_module.state_dict()
            save_dict['right_eye_d'] = self.right_eye_d_module.state_dict()
            save_dict['mouth_d'] = self.mouth_d_module.state_dict()

        torch.save(
                   save_dict,
                   path
                )

    def get_lr(self):
        return self.optimG.state_dict()['param_groups'][0]['lr']


    def select_img(self, data, size=None, name='fake', axis=2):
        if size is None:
            size = self.args.size

        if not data or len(data) == 0:
            print("Error: No images received in select_img()")
            return {}

        #Ensure last image is a tensor
        selected_img = data[-1]
        if not isinstance(selected_img, torch.Tensor):
            print("Error: Expected tensor, but got", type(selected_img))
            return {}

        # Ensure selected_img has batch dimension before resizing
        selected_img = selected_img.unsqueeze(0) if selected_img.dim() == 3 else selected_img
        selected_img = F.adaptive_avg_pool2d(selected_img, size)  # Resize to target size

        # Now call parent class `select_img()` correctly
        return super().select_img([selected_img], name, axis)  # Keep it as a list

#The Gram matrix captures the correlation between different feature maps, making it useful for: Feature correlation analysis, Feature Analysis
    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)  #This converts each feature map (channel) into a long vector (flattened spatial dimensions).
        features_t = features.transpose(1, 2)  #Transposes features to shape (n, h*w, c).
        gram = features.bmm(features_t) / (c * h * w)  #computes Gram Matrix by performing batch matrix multiplication(BMM) between features and features_t.
        return gram


    def gray_resize_for_identity(self, out, size=128):
        out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
        out_gray = out_gray.unsqueeze(1)
        out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
        return out_gray
