import pdb
from projects.DistillNeRF.models.geometry_parameterized import Geometry_P
from hydra.utils import instantiate
from itertools import chain
import time
import torch
from projects.DistillNeRF.utils.utils import Container
do_print = False

class DistillNerfModel(Geometry_P):
    """
    Continuous 3D autoencoder
    """
    def set(self, property, config, default_value):
        if property in config:
            setattr(self, property, config.get(property))
        else:
            setattr(self, property, default_value)

    def __init__(self, config):
        super().__init__(config)
        ''' depth anything, whose depth feature is used for depth prediction '''
        self.mono_depth_pretained_model = instantiate(config.mono_depth_pretained_model)
        for param in self.mono_depth_pretained_model.parameters(): param.requires_grad = False

        ''' coarse_mono_depth_estimator: predict the coarse depth as the first stage of single view encoding/lifting  '''
        self.coarse_monodepth_estimator = instantiate(config.coarse_mono_depth_estimator)

        ''' fine_mono_depth_estimator: predict the fine depth as the second stage of single view encoding/lifting  '''
        self.fine_mono_depth_estimator = instantiate(config.fine_mono_depth_estimator)

        ''' multi_view_pooling_octree_encoder: 
                predict depth candidate (second stage of single view encoding/lifting),   
                multi-view fusion
        '''
        self.multi_view_pooling_octree_encoder = instantiate(config.multi_view_pooling_octree_encoder, geometry=self)
        
        ''' decoder '''
        if 'decoder' in config.keys(): self.decoder = instantiate(config.decoder)

        ''' projector '''
        self.projector = instantiate(config.projector, geometry=self)

        self.prev_time = time.time()
        self.local_rank = 0

    def set_local_rank(self, local_rank):
        self.local_rank = local_rank
        self.multi_view_pooling_octree_encoder.set_local_rank(local_rank)
        self.coarse_monodepth_estimator.set_local_rank(local_rank)
        if 'decoder' in self.config.keys(): self.decoder.set_local_rank(local_rank)
        self.projector.set_local_rank(local_rank)

    def configure_optimizers(self): # for pytorch lightning
        opt = instantiate(self.config.optimizer,
                          params=chain(self.multi_view_pooling_octree_encoder.parameters(), self.decoder.parameters(), self.projector.parameters()))
        return opt

    def encode(self, scene_data, intermediates):
        start = time.time()
        with torch.no_grad():
            ''' get depth feature from pretrained depth model '''
            if self.mono_depth_pretained_model is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float32):
                    intermediates = self.mono_depth_pretained_model(scene_data, intermediates)
        if do_print: print("    Encoder - pretrain depth:", round(time.time()-start,6))
                    
        ''' coarse mono depth '''
        imgs = scene_data.imgs
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B*N, C, imH, imW)
        cam_classes = scene_data.cam_classes.view(B*N) if scene_data.has("cam_classes") else None
        start = time.time()
        intermediates = self.coarse_monodepth_estimator(imgs,
                intermediates=intermediates,
                classes=cam_classes
        )
        if do_print: print("    Encoder - coarse depth:", round(time.time()-start,6))

        ''' fine mono depth, and features for each view '''
        imgs = scene_data.imgs
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B*N, C, imH, imW)
        cam_classes = scene_data.cam_classes.view(B*N) if scene_data.has("cam_classes") else None
        torch.cuda.synchronize()
        start = time.time()
        out, intermediates = self.fine_mono_depth_estimator(imgs, intermediates=intermediates, classes=cam_classes)
        torch.cuda.synchronize()
        if do_print: print("    Encoder - Fine depth:", round(time.time()-start,6))

        encoded_scene, weights = out
        intermediates.set("encoded_scene", encoded_scene)

        ''' multi-view pooling octree encoder '''
        torch.cuda.synchronize()
        start = time.time()
        intermediates = self.multi_view_pooling_octree_encoder(scene_data, intermediates)
        torch.cuda.synchronize()
        if do_print: print("    Encoder - multi-view fusion:", round(time.time()-start,6))

        # delete some features not used later, to save memory
        del intermediates.encoded_scene
        del intermediates.monodepthfeat
        del intermediates.fine_depth_candidates
                         
        return intermediates

    def print_info(self):
        def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
        try:
            print("\n\n\n")
            print("Encoder:", "{:,}".format(count_parameters(self.multi_view_pooling_octree_encoder)))
            print("Projector:", "{:,}".format(count_parameters(self.projector)))
            print("Decoder:", "{:,}".format(count_parameters(self.decoder)))
            print("\n\n\n")
        except:
            print ("Parameter not countable")
    
    def forward(self, scene_data: Container):
        intermediates = Container()
        torch.cuda.synchronize()

        if True: print(f"Prep+backward time: {round(time.time()-self.prev_time,6)}")
        self.prev_time = time.time()
        meta_start = time.time()

        ''' encoder '''
        start = time.time()
        intermediates = self.encode(scene_data, intermediates)
        if True: print("Encoder:", round(time.time()-start,6))

        ''' projector '''
        start = time.time()
        intermediates = self.projector(scene_data, intermediates)
        if do_print: print("Projector:", round(time.time()-start,6))

        ''' reconstruction '''
        start = time.time()
        model_outs = self.decoder(scene_data, intermediates)
        # import pdb; pdb.set_trace()
        if do_print: print("Decoder:", round(time.time()-start,6))
        if True: print("Forward:", round(time.time()-meta_start,6))
        torch.cuda.synchronize()

        return model_outs

    @property
    def last_layer(self):
        return self.decoder.last_layer


class DistillNerfModelDecoderFree(DistillNerfModel):
    
    def forward(self, scene_data: Container):
        intermediates = Container()

        if do_print: print(f"Prep+backward time: {round(time.time()-self.prev_time,2)}")
        self.prev_time = time.time()
        meta_start = time.time()

        ''' encoder '''
        start = time.time()
        intermediates = self.encode(scene_data, intermediates)
        if do_print: print("Encoder:", round(time.time()-start,6))

        ''' projector '''
        start = time.time()
        intermediates = self.projector(scene_data, intermediates)
        if do_print: print("Projector:", round(time.time()-start,6))

        ''' decoder '''
        start = time.time()
        intermediates.set("recons", intermediates.target_2d_features)
        if do_print: print("Forward:", round(time.time()-meta_start,2))

        return intermediates
