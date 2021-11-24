import re
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

class Generator(object):
    def __init__(self, data, data_device, log_dir, hparams):
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)

        # model relative
        self.data_device = data_device
        self.seqlen = hparams.Data.seqlen
        self.n_lookahead = hparams.Data.n_lookahead
        self.data = data
        self.log_dir = log_dir
        #self.ee_cond = np.zeros((self.x.shape[0],self.ee_dim,self.x.shape[2]),dtype=np.float32) 
        self.ee_LH_idx=[(16)*3 +0,(16)*3 +1,(16)*3 +2]
        self.ee_RH_idx=[(20)*3 +0,(20)*3 +1,(20)*3 +2]
        self.ee_RF_idx=[(8)*3 +0,(8)*3 +1,(8)*3 +2]
        self.ee_LF_idx=[(4)*3 +0,(4)*3 +1,(4)*3 +2]
        self.ee_HEAD_idx = [(12)*3 +0,(12)*3 +1,(12)*3 +2]

        self.ee_dim = 5*3 # Head, LH, RH, RF, LF 순서로 가자
        # test batch
        self.test_data_loader = DataLoader(data.get_test_dataset(),
                                      batch_size=hparams.Train.batch_size,
                                      num_workers=1,
                                      shuffle=False,
                                      drop_last=True)
        self.test_batch = next(iter(self.test_data_loader))
        for k in self.test_batch:
            self.test_batch[k] = self.test_batch[k].to(self.data_device)
    
    def prepare_eecond(self, jt_data):
        # input data inside ee_cond 
        jt_data = np.swapaxes(jt_data,1,2)
        ee_cond = np.zeros((jt_data.shape[0],self.ee_dim,jt_data.shape[2]),dtype=np.float32) 
        ee_cond[:,:3,:] = jt_data[:,self.ee_HEAD_idx,:]
        ee_cond[:,(3):(3)+3,:] = jt_data[:,self.ee_LH_idx,:]
        ee_cond[:,(6):(6)+3,:] = jt_data[:,self.ee_RH_idx,:]
        ee_cond[:,(9):(9)+3,:] = jt_data[:,self.ee_RF_idx,:]
        ee_cond[:,(12):(12)+3,:] = jt_data[:,self.ee_LF_idx,:]
        ee_cond = torch.from_numpy(ee_cond)
        return ee_cond.to(self.data_device)

    def prepare_cond(self, jt_data, ctrl_data):
        nn,seqlen,n_feats = jt_data.shape
        
        jt_data = jt_data.reshape((nn, seqlen*n_feats))
        nn,seqlen,n_feats = ctrl_data.shape
        ctrl_data = ctrl_data.reshape((nn, seqlen*n_feats))
        cond = torch.from_numpy(np.expand_dims(np.concatenate((jt_data,ctrl_data),axis=1), axis=-1))
        return cond.to(self.data_device)
    
    def generate_sample_withRef(self, graph, eps_std=1.0, step=0, counter=0):
        print("generate_sample")

        batch = self.test_batch

        autoreg_all = batch["autoreg"].cpu().numpy()
        control_all = batch["control"].cpu().numpy()

        # Initialize the pose sequence with ground truth test data
        seqlen = self.seqlen
        n_lookahead = self.n_lookahead
        
        # Initialize the lstm hidden state
        if hasattr(graph, "module"):
            graph.module.init_lstm_hidden()
        else:
            graph.init_lstm_hidden()
            
        nn,n_timesteps,n_feats = autoreg_all.shape
        sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
        reference_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
        autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
        sampled_all[:,:seqlen,:] = autoreg
        
        # Loop through control sequence and generate new data
        for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
            control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
            refpose = autoreg_all[:,(i+seqlen):(i+seqlen+1),:]
            # 전체 포즈에서 end-effector condition 을 만들어야한다
            ee_cond = self.prepare_eecond(refpose.copy())

            # prepare conditioning for moglow (control + previous poses)
            cond = self.prepare_cond(autoreg.copy(), control.copy())
            
            # sample from Moglow
            #sampled_z_random = ( torch.ones((nn,63,1)) * -1.0).to(self.data_device)
            sampled = graph(z=None, cond=cond, eps_std=eps_std, reverse=True)
            sampled = sampled.cpu().numpy()[:,:,0]

            # store the sampled frame
            sampled_all[:,(i+seqlen),:] = sampled # sampled
            reference_all[:,(i+seqlen),:] = np.swapaxes(refpose,1,2)[:,:,0] # GT
            # update saved pose sequence
            autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
            
        # store the generated animations
        self.data.save_animation_withRef(control_all[:,:(n_timesteps-n_lookahead),:], sampled_all, reference_all, os.path.join(self.log_dir, f'sampled_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
    
    def generate_APD_perBatch(self, graph, eps_std=1.0, step=0, counter=0):
        print("generate_sample")

        batch = self.test_batch

        autoreg_all = batch["autoreg"].cpu().numpy()
        control_all = batch["control"].cpu().numpy()
        
        # Initialize the pose sequence with ground truth test data
        seqlen = self.seqlen
        n_lookahead = self.n_lookahead
        
        nn,n_timesteps,n_feats = autoreg_all.shape
        reference_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
        reference_all[:,seqlen:,:] = autoreg_all[:,seqlen:,:]
        nTestMotionClips = 10
        test_sampled = np.zeros((nTestMotionClips,nn,n_timesteps,n_feats))  
        #test_z_values = np.linspace(-7,7,nTestMotionClips)        
        for K in range(nTestMotionClips):
            
            sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))

            autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
            sampled_all[:,:seqlen,:] = autoreg

            # Initialize the lstm hidden state
            if hasattr(graph, "module"):
                graph.module.init_lstm_hidden()
            else:
                graph.init_lstm_hidden()
                
            sampled_z_random = graph.distribution.sample((nn,n_feats,1), eps_std, device=self.data_device)
            #sampled_z_random = ( torch.ones((nn,n_feats,1)) * test_z_values[K]).to(self.data_device)
            
            # Loop through control sequence and generate new data
            for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
                # prepare conditioning for moglow (control + previous poses)
                control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
                cond = self.prepare_cond(autoreg.copy(), control.copy())
                          
                #cond = control_all[:,(i+seqlen):(i+seqlen+1+n_lookahead),:]
                #cond = torch.from_numpy(np.swapaxes(cond,1,2)).to(self.data_device)

                # sample from Moglow
                sampled = graph(z=sampled_z_random, cond=cond, eps_std=eps_std, reverse=True)
                sampled = sampled.cpu().numpy()[:,:,0]

                # store the sampled frame
                sampled_all[:,(i+seqlen),:] = sampled
                
                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)

            # test data is different by batch 
            #if K < 5:
            #self.data.save_animation_withBVH(control_all[:,:(n_timesteps-n_lookahead),:], sampled_all, self.log_dir, f'{test_z_values[K]}_MG_SP_EE_Specify_sampled_{str(int(eps_std*100))}_{str(step//1000)}k')  
            #concatenate sampled motion
            test_sampled[K] = sampled_all

        # store the generated animations
        #self.data.save_APD_Score(control_all[:,:(n_timesteps-n_lookahead),:], test_sampled, nTestMotionClips, os.path.join(self.log_dir, f'{counter}_sampled_temp{str(int(eps_std*100))}_{str(step//1000)}k'))
        self.data.save_APD_Score_withRef(reference_all,control_all[:,:(n_timesteps-n_lookahead),:], test_sampled, nTestMotionClips, os.path.join(self.log_dir, f'{counter}_sampled_temp{str(int(eps_std*100))}_{str(step//1000)}k'))         
    
    def generate_sample(self, graph, eps_std=1.0, step=0, counter=0):
        print("generate_sample")

        batch = self.test_batch

        autoreg_all = batch["autoreg"].cpu().numpy()
        control_all = batch["control"].cpu().numpy()

        # Initialize the pose sequence with ground truth test data
        seqlen = self.seqlen
        n_lookahead = self.n_lookahead
        
        # Initialize the lstm hidden state
        if hasattr(graph, "module"):
            graph.module.init_lstm_hidden()
        else:
            graph.init_lstm_hidden()
            
        nn,n_timesteps,n_feats = autoreg_all.shape
        sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
        reference_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
        autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
        sampled_all[:,:seqlen,:] = autoreg
        
        sampled_z_random = graph.distribution.sample((nn,n_feats,1), eps_std, device=self.data_device)
        # Loop through control sequence and generate new data
        for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
            control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
            # 전체 포즈에서 end-effector condition 을 만들어야한다
            refpose = autoreg_all[:,(i+seqlen):(i+seqlen+1),:]
            ee_cond = self.prepare_eecond(refpose.copy())

            # prepare conditioning for moglow (control + previous poses)
            cond = self.prepare_cond(autoreg.copy(), control.copy())

            # sample from Moglow
            #sampled_z_random = ( torch.ones((nn,63,1)) * -1.0).to(self.data_device)
            sampled = graph(z=sampled_z_random, cond=cond, eps_std=eps_std, reverse=True)
            sampled = sampled.cpu().numpy()[:,:,0]

            # store the sampled frame
            sampled_all[:,(i+seqlen),:] = sampled # sampled
            
            # update saved pose sequence
            autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
            
        # store the generated animations
        self.data.save_animation(control_all[:,:(n_timesteps-n_lookahead),:], sampled_all, os.path.join(self.log_dir, f'HierarchyNonSp_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))              
        