import numpy as np
import pdb

class ViewGenerator():
    def __init__(self, config):
        self.config = config 

    def sample(self):
        pass 

class ViewModifier():
    def __init__(self, config):
        self.config = config 

    def modify_view(self, scene_batch):
        trans_offset = np.array([0.0,0.0,0.0])
        rots_offset = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]) 

        trans_offset += np.matmul(rots_offset, np.array([0.0,self.config.lr_trans_dist.sample(),0.0]))
        trans_offset += np.matmul(rots_offset, np.array([self.config.fb_trans_dist.sample(),0.0,0.0]))
        trans_offset += np.matmul(rots_offset, np.array([0.0,0.0,self.config.ud_trans_dist.sample()]))

        rots_offset = np.matmul(rots_offset, np.array([[1.0,0.0,0.0],[0, np.cos(np.pi/36*self.config.xtheta_dist.sample()),-np.sin(np.pi/36*self.config.xtheta_dist.sample())],[0, np.sin(np.pi/36*self.config.xtheta_dist.sample()),np.cos(np.pi/36*self.config.xtheta_dist.sample())]]))
        rots_offset = np.matmul(rots_offset, np.array([[np.cos(np.pi/36*self.config.ztheta_dist.sample()),-np.sin(np.pi/36*self.config.ztheta_dist.sample()),0.0],[np.sin(np.pi/36*self.config.ztheta_dist.sample()),np.cos(np.pi/36*self.config.ztheta_dist.sample()),0.0],[0.0,0.0,1.0]]))
        rots_offset = np.matmul(rots_offset, np.array([[np.cos(np.pi/36*self.config.ytheta_dist.sample()),0,np.sin(np.pi/36*self.config.ytheta_dist.sample())],[0.0,1.0,0.0],[-np.sin(np.pi/36*self.config.ytheta_dist.sample()),0, np.cos(np.pi/36*self.config.ytheta_dist.sample())]]))

        scene_batch.set("trans_offset", trans_offset)
        scene_batch.set("rots_offset", rots_offset)
        return scene_batch

