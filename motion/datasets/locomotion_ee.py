from operator import gt
import os
import numpy as np
from numpy.core.numeric import extend_all

import motion
from .motion_data_ee import MotionDataset, TestDataset
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from visualization.plot_animation import plot_animation, plot_animation_withRef,save_animation_BVH
from scipy.spatial.distance import pdist
from sklearn.metrics import mean_squared_error
from scipy.spatial.transform import Rotation as R

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

class LocomotionEE():

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
        
        self.scaler = scaler
        self.frame_rate = hparams.Data.framerate
        self.ee_idx =[(12)*3+0,(12)*3+1,(12)*3+2, 
        (16)*3+0,(16)*3+1,(16)*3+2,
        (20)*3+0,(20)*3+1,(20)*3+2]
        # Create pytorch data sets
        self.train_dataset = MotionDataset(train_data[:,:,-3:], train_data[:,:,:-3], hparams.Data.seqlen, hparams.Data.n_lookahead, hparams.Data.dropout)    
        self.test_dataset = TestDataset(all_test_data[:,:,-3:], all_test_data[:,:,:-3])
        self.validation_dataset = MotionDataset(validation_data[:,:,-3:], validation_data[:,:,:-3], hparams.Data.seqlen, hparams.Data.n_lookahead, hparams.Data.dropout)
        self.seqlen = hparams.Data.seqlen
        self.n_x_channels = all_test_data.shape[2]-3
        self.n_cond_channels = 3 #self.n_x_channels*hparams.Data.seqlen + 3*(hparams.Data.seqlen + 1 + hparams.Data.n_lookahead)

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

    def calc_worldjoints_withRef(self, clip, refclip):
        
        rot = R.from_quat([0,0,0,1])
        translation = np.array([[0,0,0]])
        translations = np.zeros((clip.shape[0],3))
        
        joints, root_dx, root_dz, root_dr = clip[:,:-3], clip[:,-3], clip[:,-2], clip[:,-1]
        joints_ref = refclip[:,:-3]

        joints = joints.reshape((len(joints), -1, 3))
        joints_ref = joints_ref.reshape((len(joints_ref), -1, 3))
        for i in range(len(joints)):
            joints[i,:,:] = rot.apply(joints[i])
            joints[i,:,0] = joints[i,:,0] + translation[0,0]
            joints[i,:,2] = joints[i,:,2] + translation[0,2]
            
            joints_ref[i,:,:] = rot.apply(joints_ref[i])
            joints_ref[i,:,0] = joints_ref[i,:,0] + translation[0,0]
            joints_ref[i,:,2] = joints_ref[i,:,2] + translation[0,2]
            
            
            rot = R.from_rotvec(np.array([0,-root_dr[i],0])) * rot
            translation = translation + rot.apply(np.array([root_dx[i], 0, root_dz[i]]))
            translations[i,:] = translation
        
        return joints, joints_ref

    def save_APD_Score_withRef(self, reference_data, control_data, K_motion_data, totalClips,filename):
        np.savez("../data/results/test_Loco_Full/40_APD_withRef_testdata.npz", clips=K_motion_data)
        
        #K_motion_data = np.load("../data/results/test_Loco_Full/2.0_MG_SPEE_T1_100_0k_APD_withRef_testdata.npz")['clips'].astype(np.float32) 
        #K_motion_data = np.load("../data/results/test_Loco_Full_nonSp/2.0_MG_SPEE_T1_100_0k_APD_withRef_testdata.npz")['clips'].astype(np.float32) 
        
        
        
        K, nn, ntimesteps, feature = K_motion_data.shape
        total_APD_score = np.zeros(nn)
        total_ADE_score = np.zeros(nn)
        total_FDE_score = np.zeros(nn)
        total_EED_score = np.zeros(nn)
        if totalClips != K:
            print("wrong! different motions")
        else :

            for nBatch in range(nn):
                if nBatch != 31 and nBatch != 69:
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
                    
                    motion_clip = anim_clip[...,:-3] /26.2
                    gt_clip = ref_clip[...,:-3] /26.2
                    

                    
                    # apd_score = 0
                    # ade_score = 0
                    # fde_score = 0
                    # eed_score = 0
                    # for i in range(K):
                    #     joints, joints_ref = self.calc_worldjoints_withRef(ref_clip[i],anim_clip[i])
                    #     joints = joints.reshape(joints.shape[0],-1)
                    #     joints_ref = joints_ref.reshape(joints_ref.shape[0],-1)

                    #     motion_clip = joints /26.2
                    #     gt_clip = joints_ref /26.2
                    
                    motion_clip = motion_clip[:,-70:,:]
                    gt_clip = gt_clip[:,-70:,:]
                    # get score
                    apd_score = self.calculate_APD(motion_clip)
                    ade_score = self.calculate_ADE(gt_clip,motion_clip)
                    eed_score = self.caclulate_EED(gt_clip,motion_clip)
                    #fde_score += self.calculate_FDE(gt_clip,motion_clip)
                    
                    total_APD_score[nBatch] = np.mean(apd_score)
                    total_ADE_score[nBatch] = np.mean(ade_score)
                    #total_FDE_score[nBatch] = np.mean(fde_score)
                    total_EED_score[nBatch] = np.mean(eed_score)

            total_APD_score = np.delete(total_APD_score,[31,69],0)
            total_ADE_score = np.delete(total_ADE_score,[31,69],0)
            total_EED_score = np.delete(total_EED_score,[31,69],0)

            print(f'APD of_{nn}_motion:_{total_APD_score.shape}_:{total_APD_score}_mean:{np.mean(total_APD_score)}')    
            np.savez(filename + "_APD_score.npz", clips=total_APD_score)
            print(f'ADE of_{nn}_motion:_{total_ADE_score.shape}_:{total_ADE_score}_mean:{np.mean(total_ADE_score)}')    
            np.savez(filename + "_ADE_score.npz", clips=total_ADE_score)
            print(f'EED of_{nn}_motion:_{total_EED_score.shape}_:{total_EED_score}_mean:{np.mean(total_EED_score)}')    
            np.savez(filename + "_EED_score.npz", clips=total_EED_score)

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
    
    def caclulate_EED(self, gt_clip, motion_clip):
        
        

        motion_clip_ee = motion_clip[:,:,self.ee_idx]
        gt_clip_ee = gt_clip[:,:,self.ee_idx]

        # k number's l2 (diff) 
        #diff = motion_clip_ee - gt_clip_ee
        #dist = np.linalg.norm(diff, axis=1).mean(axis=-1)
        #dist.mean()
        motion_clip_ee = np.reshape(motion_clip_ee,(motion_clip_ee.shape[0],-1))
        gt_clip_ee = np.reshape(gt_clip_ee,(gt_clip_ee.shape[0],-1))
        
        root_MSE = mean_squared_error(motion_clip_ee,gt_clip_ee,squared=False)
        return root_MSE

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
            
    def save_animation_withBVH(self, control_data, motion_data, filename):
        animation_data = np.concatenate((motion_data,control_data), axis=2)
        anim_clip = inv_standardize(animation_data, self.scaler)
        np.savez(filename + ".npz", clips=anim_clip)
        n_clips = min(self.n_test, anim_clip.shape[0])
        for i in range(0,n_clips):
            filename_ = f'{filename}_{str(i)}.mp4'
            print('writing:' + filename_)
            parents = np.array([0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]) - 1
            plot_animation(anim_clip[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=60)
            save_animation_BVH(anim_clip[i,self.seqlen:,:],parents,filename_)

    def n_channels(self):
        return self.n_x_channels, self.n_cond_channels
        
    def get_train_dataset(self):
        return self.train_dataset
        
    def get_test_dataset(self):
        return self.test_dataset

    def get_validation_dataset(self):
        return self.validation_dataset
        
		