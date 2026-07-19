from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.exp239_248_diverse_score_sources.config import SECOND_SEED
from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle, ScoreResult
from experiments.exp239_248_diverse_score_sources.foundation import crossfit_plan
from experiments.exp239_248_diverse_score_sources.neural_common import set_deterministic
from experiments.exp239_248_diverse_score_sources.source_base import abstained_result


class _Coupling(nn.Module):
    def __init__(self,dimension:int,parity:int)->None:
        super().__init__();mask=(torch.arange(dimension)%2==parity).float();self.register_buffer("mask",mask);self.net=nn.Sequential(nn.Linear(dimension,32),nn.GELU(),nn.Linear(32,32),nn.GELU(),nn.Linear(32,dimension*2))
    def forward(self,x:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        masked=x*self.mask;s,t=self.net(masked).chunk(2,dim=1);s=1.5*torch.tanh(s)*(1-self.mask);t=t*(1-self.mask);y=masked+(1-self.mask)*(x*torch.exp(s)+t);return y,s.sum(1)


class _RealNVP(nn.Module):
    def __init__(self,dimension:int)->None:
        super().__init__();self.layers=nn.ModuleList([_Coupling(dimension,index%2) for index in range(4)])
    def nll(self,x:torch.Tensor)->torch.Tensor:
        logdet=torch.zeros(len(x),device=x.device);z=x
        for layer in self.layers:z,delta=layer(z);logdet+=delta
        return 0.5*(z*z).sum(1)+0.5*z.shape[1]*np.log(2*np.pi)-logdet


def _fit_flow(reference:np.ndarray,seed:int,epochs:int,device:str)->tuple[_RealNVP,np.ndarray,np.ndarray]:
    set_deterministic(seed);center=np.median(reference,axis=0);scale=1.4826*np.median(np.abs(reference-center),axis=0)+1e-4;normalized=((reference-center)/scale).astype(np.float32);model=_RealNVP(reference.shape[1]).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);tensor=torch.as_tensor(normalized,device=device)
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad();loss=model.nll(tensor).mean();loss.backward();optimizer.step()
    return model.eval(),center,scale


def _flow_scores(model:_RealNVP,rows:np.ndarray,center:np.ndarray,scale:np.ndarray,device:str)->np.ndarray:
    values=torch.as_tensor(((rows-center)/scale).astype(np.float32),device=device)
    with torch.no_grad():return model.nll(values).cpu().numpy()


class NonlinearNormalDensity:
    name="nonlinear_normal_density";minimum_train=50
    def __init__(self,epochs:int=40,seeds:tuple[int,...]|None=None,device:str="cpu")->None:self.epochs=int(epochs);self.seeds=seeds;self.device=device

    def score_from_embeddings(self,train:np.ndarray,test:np.ndarray,seed:int)->ScoreResult:
        if len(train)<self.minimum_train:
            dummy=DatasetBundle("embedding",np.empty((len(train),1)),np.empty((len(test),1)),1);return abstained_result(self.name,dummy,self.minimum_train)
        seeds=self.seeds or (int(seed),SECOND_SEED);train_by_seed=[];test_by_seed=[];test_typicality=[]
        for current_seed in seeds:
            oof=np.empty(len(train))
            for fold,(fit,query) in enumerate(crossfit_plan(len(train),seed=current_seed)):
                model,center,scale=_fit_flow(train[fit],current_seed+fold,self.epochs,self.device);oof[query]=_flow_scores(model,train[query],center,scale,self.device)
            model,center,scale=_fit_flow(train,current_seed,self.epochs,self.device);test_nll=_flow_scores(model,test,center,scale,self.device);train_nll=_flow_scores(model,train,center,scale,self.device);train_by_seed.append(oof);test_by_seed.append(test_nll);test_typicality.append(np.abs(test_nll-np.median(train_nll)))
        return ScoreResult(self.name,np.median(train_by_seed,axis=0),np.median(test_by_seed,axis=0),False,None,{"seeds":list(seeds),"epochs":self.epochs,"layers":4,"width":32,"device":self.device,"test_used_for_fit":False,"cross_fitted":True,"self_reference_violations":0,"contains_test_labels":False},artifacts={"test_typicality":np.median(test_typicality,axis=0),"train_nll_by_seed":np.asarray(train_by_seed),"test_nll_by_seed":np.asarray(test_by_seed)})

    def score(self,bundle:DatasetBundle,seed:int,artifact_dir:Path)->ScoreResult:
        path=Path(artifact_dir)/"masked_patch_embeddings.npz"
        if not path.exists():raise FileNotFoundError(f"missing frozen Exp247 embeddings: {path}")
        with np.load(path) as archive:return self.score_from_embeddings(archive["train_embeddings"],archive["test_embeddings"],seed)
