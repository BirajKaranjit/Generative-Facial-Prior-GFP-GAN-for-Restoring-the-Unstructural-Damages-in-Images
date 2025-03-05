#ModelTrainer.py
import torch
import tqdm
import os
import time
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP

class ModelTrainer:

    def __init__(self,args):
        super(ModelTrainer, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.old_lr = args.lr

    def print_generator_layers(self):
        """Print all layers of the generator to identify which layers to freeze."""
        print("Generator Layers:")
        for name, param in self.netG.named_parameters():
            print(f"{name}: {param.shape}")

    def freeze_models(self, frozen_params):
        """
        Freeze specific layers of the generator.
        :param frozen_params: List of layer name prefixes to freeze (e.g., ["conv_body", "condition_scale"]).
        """
        for name, param in self.netG.named_parameters():
            # Check if the parameter name starts with any prefix in frozen_params
            if any(name.startswith(layer_prefix) for layer_prefix in frozen_params):
                param.requires_grad = False  # Freeze the layer
            else:
                param.requires_grad = True  # Unfreeze the rest


    # ## ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== =====

    def train_network(self, train_loader, test_loader):

      #clear gpu cache at the start of a training loop
      torch.cuda.empty_cache()
      counter = 0
      loss_dict = {}
      acc_num = 0
      mn_loss = float('inf')

      steps = 0
      begin_it = 0
      #Lists to store training and validation losses for each batch
      train_losses = []  # Training loss for each batch
      val_losses = []    # Validation loss for each batch
      val_steps = []     # Steps at which validation was performed

      if self.args.pretrain_path:
          try:
              begin_it = int(self.args.pretrain_path.split('/')[-1].split('-')[0]) + 1
              steps = int(self.args.pretrain_path.split('/')[-1].split('-')[1].replace('.pth', '')) + 1
          except:
              steps = 0
              begin_it = 0
          # print("current steps: %d | one epoch steps: %d" % (steps, self.args.mx_data_length))
      if not self.args.apply_begin_it:
          begin_it = 0
          steps = 0

      # Print generator layers to identify which layers to freeze
      # self.print_generator_layers()  ##uncomment only if you want to see what layers actually exists

      # Freeze initial layers of the generator
      frozen_params = ["conv_body_first","conv_body_down","condition_scale","condition_shift","stylegan_decoder.style_mlp","stylegan_decoder.style_conv1","stylegan_decoder.style_convs","stylegan_decoder.to_rgbs","stylegan_decoder.final_rgb","stylegan_decoder.final_conv1","stylegan_decoder.final_conv2"]
      self.freeze_models(frozen_params)

      # Training loop
      try:  #this 'try' and 'except' is added to clear the GPU cache after the code fails
        for epoch in range(begin_it, self.args.max_epoch):
            #set models to TRAINING MODE
            self.netG.train()

            # Wrap the training loop with tqdm
            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.max_epoch}", unit="batch") as pbar:
                for ii, data in enumerate(pbar):
                    lq, hq = data

                    # Freeze discriminator weights  #NOTE: '.parametes()' -> Returns an iterator over module parameters, yielding only the parameter tensors (no names).
                    for param in self.netD.parameters():
                        param.requires_grad = False


                    # Unfreeze generator weights (only for the UPSAMPING layers to fine-tune)
                    for name, param in self.netG.named_parameters():
                        if name.startswith("final_body_up"):
                            param.requires_grad = True  # Unfreeze final_body_up layers
                            # print(f"{name}: requires_grad = {param.requires_grad}")
                        else:
                            param.requires_grad = False  # Freeze all other layers

                    tstart = time.time()
                    self.run_single_step(data, steps)
                    telapsed = time.time() - tstart

                    losses = self.get_latest_losses()
                    # print(f"Losses dictionary at step {steps}: {losses}")  # Debug print
                    for key, val in losses.items():   # Accumulate losses
                        # print(f"Accumulating loss for {key}: {val}")  # Debug print
                        loss_dict[key] = loss_dict.get(key, 0) + val.mean().item()

                    counter += 1

                    # Log losses for each batch
                    if steps % self.args.print_interval == 0 and self.args.rank == 0:
                        for key, val in loss_dict.items():
                            loss_dict[key] /= counter
                            # print(f"Loss after averaging {key}: {loss_dict[key]}")  # Debug print

                        lr_rate = self.get_lr()
                        # lr_rate = 0.005  #i changed on March 1
                        print_dict = {"time": telapsed, **loss_dict}
                        self.vis.print_current_errors(epoch, steps, print_dict, telapsed)
                        self.vis.plot_current_errors(print_dict, steps)

                        # Reset loss_dict and counter for the next interval
                        loss_dict = {}
                        counter = 0

                     # Store training loss for the batch
                    train_losses.append(float(losses.get("loss", 0)))
                    steps += 1

                    # Update the progress bar with the current batch's loss
                    current_batch_loss = losses.get("g_pixloss", 0).item()  # Get the current batch's loss
                    pbar.set_postfix({"Loss": current_batch_loss})  # Update tqdm progress bar with current loss

            # Save model checkpoint at the end of each epoch
            if self.args.rank == 0:
                self.saveParameters(os.path.join(self.args.checkpoint_path, "%03d-%08d.pth" % (epoch + 1, steps)))
                self.current_epoch, self.current_step = epoch + 1, steps

            # Evaluate on validation set at the end of each epoch
            if test_loader and self.args.eval:
                val_loss = self.evalution(test_loader, steps, epoch)
                val_losses.append(val_loss['loss'].item())  # Store validation loss
                val_steps.append(steps)  # Store the step at which validation was performed
                if self.args.early_stop:
                    acc_num, mn_loss, stop_flag = self.early_stop_wait(self.get_loss_from_val(val_loss), acc_num, mn_loss)
                    if stop_flag:
                        return

      except Exception as e:
        # Clear GPU cache if an exception occurs
        print(f"An error occurred: {e}")
        torch.cuda.empty_cache()
        raise  # Re-raise the exception after clearing the cache
      finally:
        # Clear GPU cache at the end of training
        torch.cuda.empty_cache()
        # Plot training and validation loss curves
        if self.args.rank == 0:
            self.plot_loss_curves(train_losses, val_losses, val_steps)

      if self.args.rank == 0:
          self.vis.close()


    # def early_stop_wait(self,loss,acc_num,mn_loss):

    #     if self.args.rank == 0:
    #         if loss < mn_loss:
    #             mn_loss = loss
    #             cmd_one = 'cp -r %s %s'%(os.path.join(
    #                     self.args.checkpoint_path,"%03d-%08d.pth"%(
    #                             self.current_epoch,self.current_step)),
    #                                             os.path.join(self.args.checkpoint_path,'final.pth'))
    #             done_one = subprocess.Popen(cmd_one,stdout=subprocess.PIPE,shell=True)
    #             done_one.wait()
    #             acc_num = 0
    #         else:
    #             acc_num += 1

    #         if self.args.dist:

    #             if acc_num > self.args.stop_interval:
    #                 signal = torch.tensor([0]).cuda()
    #             else:
    #                 signal = torch.tensor([1]).cuda()
    #     else:
    #         if self.args.dist:
    #             signal = torch.tensor([1]).cuda()

    #     if self.args.dist:
    #         dist.all_reduce(signal)
    #         value = signal.item()
    #         if value >= int(os.environ.get("WORLD_SIZE","1")):
    #             dist.all_reduce(torch.tensor([0]).cuda())
    #             return acc_num,mn_loss,False
    #         else:
    #             return acc_num,mn_loss,True

    #     else:
    #         if acc_num > self.args.stop_interval:
    #             return acc_num,mn_loss,True
    #         else:
    #             return acc_num,mn_loss,False

    def run_single_step(self,data,steps):
        data = self.process_input(data)
        # self.netG.train()
        # lq,gt = data #do i have to do this, let's see if it works
        # self.run_discriminator_one_step(data,steps)
        self.run_generator_one_step(data,steps)

    def convert_img(img,unit=False):  #imported from utils.py

        img = (img + 1) * 0.5
        if unit:
            return torch.clamp(img*255+0.5,0,255)

        return torch.clamp(img,0,1)


    # def select_img(self,data,name='fake',axis=2):
    #     if data is None:
    #         return None
    #     cat_img = []
    #     for v in data:
    #         cat_img.append(v.detach().cpu())

    #     cat_img = torch.cat(cat_img,-1)
    #     cat_img = torch.cat(torch.split(cat_img,1,dim=0),axis)[0]

    #     return {name:convert_img(cat_img)}



    ##################################################################
    # Helper functions
    ##################################################################

    def get_loss_from_val(self,loss):
        return loss['loss']

    def get_show_inp(self,data):
        if not isinstance(data,list):
            return [data]
        return data

    def use_ddp(self,model,dist=True,use_params=False):
        if model is None:
            return None,None
        if not dist:
            return model,model
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) #
        # model = DDP(model,broadcast_buffers=False,find_unused_parameters=True) # find_unused_parameters
        model = DDP(model,
                    broadcast_buffers=False,
                    find_unused_parameters=use_params
                    )
        model_on_one_gpu = model.module
        return model,model_on_one_gpu

    def process_input(self,data):

        if torch.cuda.is_available():
            if isinstance(data,list):
                data = [x.cuda() for x in data]
            else:
                data = data.cuda()
        return data