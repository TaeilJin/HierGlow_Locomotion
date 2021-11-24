from operator import gt
import os
import numpy as np

import motion
from .motion_data import MotionDataset, TestDataset
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from visualization.plot_animation import plot_animation, plot_animation_withRef,save_animation_BVH
from scipy.spatial.distance import pdist

from motion.datasets import motion_data

def mirror_data(data):
    aa = data.copy()
    aa[:,:,3:15]=data[:,:,15:27]
    aa[:,:,3:15:3]=-data[:,:,15:27:3]
    aa[:,:,15:27]=data[:,:,3:15]
    aa[:,:,15:27:3]=-data[:,:,3:15:3]
    aa[:,:,39:51]=data[:,:,51:63]
    aa[:,:,39:51:3]=-data[:,:,51:63:3]
    aa[:,:,51:63]=data[:,:,39:51]
    aa[:,:,51:63:3]=-data[:,:,39:51:3]
    aa[:,:,63]=-data[:,:,63]
    aa[:,:,65]=-data[:,:,65]
    return aa

def reverse_time(data):
    aa = data[:,-1::-1,:].copy()
    aa[:,:,63] = -aa[:,:,63]
    aa[:,:,64] = -aa[:,:,64]
    aa[:,:,65] = -aa[:,:,65]
    return aa

def inv_standardize(data, scaler):      
    shape = data.shape
    flat = data.reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled        

def fit_and_standardize(data):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(shape)
    return scaled, scaler

def standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.transform(flat).reshape(shape)
    return scaled
  
def create_synth_test_data(n_frames, nFeats, scaler):

    syth_data = np.zeros((7,n_frames,nFeats))
    lo_vel = 1.0
    hi_vel = 2.5
    lo_r_vel = 0.08
    hi_r_vel = 0.08
    syth_data[0,:,63:65] = 0
    syth_data[1,:,63] = lo_vel
    syth_data[2,:,64] = lo_vel
    syth_data[3,:,64] = hi_vel
    syth_data[4,:,64] = -lo_vel
    syth_data[5,:,64] = lo_vel
    syth_data[5,:,65] = lo_r_vel
    syth_data[6,:,64] = hi_vel
    syth_data[6,:,65] = hi_r_vel
    syth_data = standardize(syth_data, scaler)
    syth_data[:,:,:63] = np.zeros((syth_data.shape[0],syth_data.shape[1],63))
    return syth_data.astype(np.float32)

class Locomotion():

    def __init__(self, hparams, is_training):
    
        data_root = hparams.Dir.data_root

        #load data
        train_series = np.load(os.path.join(data_root, 'all_locomotion_train_'+str(hparams.Data.framerate)+'fps.npz'))
        train_data = train_series['clips'].astype(np.float32)
        test_series = np.load(os.path.join(data_root, 'all_locomotion_test_'+str(hparams.Data.framerate)+'fps.npz'))
        test_data = test_series['clips'].astype(np.float32)
        
        print("input_data: " + str(train_data.shape))
        print("test_data: " + str(test_data.shape))

        # Split into train and val sets
        validation_data = train_data[-100:,:,:]
        train_data = train_data[:-100,:,:]
        
        # Data augmentation
        if hparams.Data.mirror:
            mirrored = mirror_data(train_data)
            train_data = np.concatenate((train_data,mirrored),axis=0)
            
        if hparams.Data.reverse_time:
            rev = reverse_time(train_data)
            train_data = np.concatenate((train_data,rev),axis=0)

        # Standardize
        train_data, scaler = fit_and_standardize(train_data)        
        validation_data = standardize(validation_data, scaler)
        all_test_data = standardize(test_data, scaler)
        synth_data2 = create_synth_test_data(test_data.shape[1], test_data.shape[2], scaler)
        all_test_data = np.concatenate((all_test_data, synth_data2), axis=0)
        self.n_test = all_test_data.shape[0]
        n_tiles = 1+hparams.Train.batch_size//self.n_test
        all_test_data = np.tile(all_test_data.copy(), (n_tiles,1,1))
        
        # np.savez("train.npz", clips=train_data)
        # np.savez("test.npz", clips=validation_data)
        # np.savez("vaild.npz", clips=all_test_data)

        # train_data = np.load('train.npz')['clips'].astype(np.float32)
        # validation_data = np.load('vaild.npz')['clips'].astype(np.float32)
        # all_test_data = np.load('test.npz')['clips'].astype(np.float32)
        
        self.scaler = scaler
        self.frame_rate = hparams.Data.framerate
		                
        # Create pytorch data sets
        self.train_dataset = MotionDataset(train_data[:,:,-3:], train_data[:,:,:-3], hparams.Data.seqlen, hparams.Data.n_lookahead, hparams.Data.dropout)    
        self.test_dataset = TestDataset(all_test_data[:,:,-3:], all_test_data[:,:,:-3])
        self.validation_dataset = MotionDataset(validation_data[:,:,-3:], validation_data[:,:,:-3], hparams.Data.seqlen, hparams.Data.n_lookahead, hparams.Data.dropout)
        self.seqlen = hparams.Data.seqlen
        self.n_x_channels = all_test_data.shape[2]-3
        self.n_cond_channels = self.n_x_channels*hparams.Data.seqlen + 3*(hparams.Data.seqlen + 1 + hparams.Data.n_lookahead)

    def save_APD_Score(self, control_data, K_motion_data, totalClips,filename):
        np.savez(filename + "_APD_testdata.npz", clips=K_motion_data)
        #K_motion_data = np.load("../data/results/locomotion/MG/log_20211103_1638/0_sampled_temp100_0k_APD_testdata.npz")['clips'].astype(np.float32)
        K, nn, ntimesteps, feature = K_motion_data.shape
        total_APD_score = np.zeros(nn)
        if totalClips != K:
            print("wrong! different motions")
        else :
            for nBatch in range(nn):
                k_motion_data = K_motion_data[:,nBatch,...]
                batch_control_data = control_data[nBatch:nBatch+1,...]
                k_control_data = np.repeat(batch_control_data,K,axis=0)

                apd_score = self.calculate_APD(k_control_data,k_motion_data)

                total_APD_score[nBatch] = apd_score
            print(f'APD of_{nn}_motion:_{total_APD_score.shape}_:{total_APD_score}_mean:{np.mean(total_APD_score)}')    
            np.savez(filename + "_APD_score.npz", clips=total_APD_score)

    def save_APD_Score_withRef(self, reference_data, control_data, K_motion_data, totalClips,filename):
        np.savez(filename + "_APD_withRef_testdata.npz", clips=K_motion_data)
        #K_motion_data = np.load("/root/home/project/MG_nonEE_66/-7.0_MG_SP_EE_Specify_sampled_100_0k.npz")['clips'].astype(np.float32)
        K, nn, ntimesteps, feature = K_motion_data.shape
        total_APD_score = np.zeros(nn)
        total_ADE_score = np.zeros(nn)
        total_FDE_score = np.zeros(nn)

        if totalClips != K:
            print("wrong! different motions")
        else :
            for nBatch in range(nn):
                k_motion_data = K_motion_data[:,nBatch,...]
                batch_reference_data = reference_data[nBatch:nBatch+1,...]
                batch_control_data = control_data[nBatch:nBatch+1,...]
                k_control_data = np.repeat(batch_control_data,K,axis=0)
                k_gt_data = np.repeat(batch_reference_data,K,axis=0)

                # after scaler
                animation_data = np.concatenate((k_motion_data,k_control_data), axis=2)
                anim_clip = inv_standardize(animation_data, self.scaler)
                gt_data = np.concatenate((k_gt_data,k_control_data), axis=2)
                ref_clip = inv_standardize(gt_data, self.scaler)
                motion_clip = anim_clip[:,self.seqlen:,:-3]/20.0
                gt_clip = ref_clip[:,self.seqlen:,:-3]/20.0

                # get score
                apd_score = self.calculate_APD(motion_clip)
                ade_score = self.calculate_ADE(gt_clip,motion_clip)
                fde_score = self.calculate_FDE(gt_clip,motion_clip)
                total_APD_score[nBatch] = apd_score
                total_ADE_score[nBatch] = ade_score
                total_FDE_score[nBatch] = fde_score

            print(f'APD of_{nn}_motion:_{total_APD_score.shape}_:{total_APD_score}_mean:{np.mean(total_APD_score)}')    
            np.savez(filename + "_APD_score.npz", clips=total_APD_score)
            print(f'ADE of_{nn}_motion:_{total_ADE_score.shape}_:{total_ADE_score}_mean:{np.mean(total_ADE_score)}')    
            np.savez(filename + "_ADE_score.npz", clips=total_ADE_score)
            print(f'FDE of_{nn}_motion:_{total_FDE_score.shape}_:{total_FDE_score}_mean:{np.mean(total_FDE_score)}')    
            np.savez(filename + "_FDE_score.npz", clips=total_FDE_score)

    def calculate_FDE(self, gt_clip, motion_clip):
        
        diff = motion_clip - gt_clip # (K,70,63)
        
        # k number's l2 (diff) 
        dist = np.linalg.norm(diff, axis=1)[:,-1]
        return dist.min()


    def calculate_ADE(self, gt_clip, motion_clip):
        
        diff = motion_clip-gt_clip # (K,70,63)
        
        # k number's l2 (diff) 
        dist = np.linalg.norm(diff, axis=1).mean(axis=-1)
        return dist.min()
           

    def calculate_APD(self, motion_clip):
        
        motion_clip = np.reshape(motion_clip,(motion_clip.shape[0],-1))
        
        dist = pdist(motion_clip)

        apd = dist.mean().item()

        # #check
        # apd =0
        # n_clips = min(self.n_test, anim_clip.shape[0])
        # for i in range(0,n_clips):
        #     filename_ = f'test_{str(i)}.mp4'
        #     print('writing:' + filename_)
        #     parents = np.array([0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]) - 1
        #     plot_animation(anim_clip[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=60)

        return (apd)

    def save_animation_withRef(self, control_data, motion_data, refer_data, filename):
        animation_data = np.concatenate((motion_data,control_data), axis=2)
        reference_data = np.concatenate((refer_data,control_data), axis=2)

        anim_clip = inv_standardize(animation_data, self.scaler)
        ref_clip = inv_standardize(reference_data,self.scaler)
        np.savez(filename + ".npz", clips=anim_clip)
        n_clips = min(self.n_test, anim_clip.shape[0])
        for i in range(0,n_clips):
            filename_ = f'{filename}_{str(i)}.mp4'
            print('writing:' + filename_)
            parents = np.array([0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]) - 1
            plot_animation_withRef(anim_clip[i,self.seqlen:,:],ref_clip[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=60)
            
    def save_animation_withRef_withBVH(self, control_data, motion_data, refer_data, logdir, filename):
        animation_data = np.concatenate((motion_data,control_data), axis=2)
        reference_data = np.concatenate((refer_data,control_data), axis=2)

        anim_clip = inv_standardize(animation_data, self.scaler)
        ref_clip = inv_standardize(reference_data,self.scaler)
        np.savez(filename + ".npz", clips=anim_clip)
        n_clips = min(self.n_test, anim_clip.shape[0])
        for i in range(0,n_clips):
            filename_ = f'{logdir}/{str(i)}_{filename}.mp4'
            print('writing:' + filename_)
            parents = np.array([0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]) - 1
            plot_animation_withRef(anim_clip[i,self.seqlen:,:],ref_clip[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=60)
            save_animation_BVH(anim_clip[i,self.seqlen:,:],parents,filename_)

    # def save_animation_withRef_withBVH(self, control_data, motion_data, refer_data, filename):
    #     animation_data = np.concatenate((motion_data,control_data), axis=2)
    #     reference_data = np.concatenate((refer_data,control_data), axis=2)

    #     anim_clip = inv_standardize(animation_data, self.scaler)
    #     ref_clip = inv_standardize(reference_data,self.scaler)
    #     np.savez(filename + ".npz", clips=anim_clip)
    #     n_clips = min(self.n_test, anim_clip.shape[0])
    #     for i in range(0,n_clips):
    #         filename_ = f'{filename}_{str(i)}.mp4'
    #         print('writing:' + filename_)
    #         parents = np.array([0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]) - 1
    #         plot_animation_withRef(anim_clip[i,self.seqlen:,:],ref_clip[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=60)
    #         save_animation_BVH(anim_clip[i,self.seqlen:,:],parents,filename_)

    def save_animation(self, control_data, motion_data, filename):
        animation_data = np.concatenate((motion_data,control_data), axis=2)
        anim_clip = inv_standardize(animation_data, self.scaler)
        np.savez(filename + ".npz", clips=anim_clip)
        n_clips = min(self.n_test, anim_clip.shape[0])
        for i in range(0,n_clips):
            filename_ = f'{filename}_{str(i)}.mp4'
            print('writing:' + filename_)
            parents = np.array([0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]) - 1
            plot_animation(anim_clip[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=60)

    def save_animation_withBVH(self, control_data, motion_data, logdir, filename):
        animation_data = np.concatenate((motion_data,control_data), axis=2)
        anim_clip = inv_standardize(animation_data, self.scaler)
        np.savez(filename + ".npz", clips=anim_clip)
        n_clips = min(self.n_test, anim_clip.shape[0])
        for i in range(0,n_clips):
            filename_ = f'{logdir}/{str(i)}_{filename}.mp4'
            print('writing:' + filename_)
            parents = np.array([0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]) - 1
            plot_animation(anim_clip[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=60)
            save_animation_BVH(anim_clip[i,self.seqlen:,:],parents,filename_)


    # def save_animation_withBVH(self, control_data, motion_data, filename):
    #     animation_data = np.concatenate((motion_data,control_data), axis=2)
    #     anim_clip = inv_standardize(animation_data, self.scaler)
    #     np.savez(filename + ".npz", clips=anim_clip)
    #     n_clips = min(self.n_test, anim_clip.shape[0])
    #     for i in range(0,n_clips):
    #         filename_ = f'{filename}_{str(i)}.mp4'
    #         print('writing:' + filename_)
    #         parents = np.array([0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]) - 1
    #         plot_animation(anim_clip[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=60)
    #         save_animation_BVH(anim_clip[i,self.seqlen:,:],parents,filename_)

    def n_channels(self):
        return self.n_x_channels, self.n_cond_channels
        
    def get_train_dataset(self):
        return self.train_dataset
        
    def get_test_dataset(self):
        return self.test_dataset

    def get_validation_dataset(self):
        return self.validation_dataset
        
		