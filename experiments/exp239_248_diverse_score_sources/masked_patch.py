from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from experiments.exp239_248_diverse_score_sources.config import SECOND_SEED
from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle, ScoreResult
from experiments.exp239_248_diverse_score_sources.foundation import crossfit_plan
from experiments.exp239_248_diverse_score_sources.neural_common import MaskedAutoencoder, prepare_rows, set_deterministic, tensor_rows
from experiments.exp239_248_diverse_score_sources.source_base import abstained_result


def _mask(values: torch.Tensor, seed: int, offset_fraction: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    length=values.shape[-1]; width=max(1,length//4)
    if offset_fraction is None:
        generator=np.random.default_rng(seed); starts=generator.integers(0,max(1,length-width+1),size=len(values))
    else:
        starts=np.full(len(values),min(length-width,int(offset_fraction*max(1,length-width))),dtype=int)
    mask=torch.zeros_like(values)
    for index,start in enumerate(starts): mask[index,:,int(start):int(start)+width]=1.0
    return values*(1.0-mask),mask


def _fit_model(rows: np.ndarray, seed: int, epochs: int, device: str) -> MaskedAutoencoder:
    set_deterministic(seed); model=MaskedAutoencoder(rows.shape[1]).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=2e-3); target=tensor_rows(rows,device)
    model.train()
    for epoch in range(int(epochs)):
        masked,mask=_mask(target,seed+epoch); optimizer.zero_grad(); reconstruction,_=model(masked); loss=((reconstruction-target)**2*mask).sum()/mask.sum().clamp_min(1.0); loss.backward(); optimizer.step()
    return model.eval()


def _model_outputs(model: MaskedAutoencoder, rows: np.ndarray, device: str) -> tuple[np.ndarray,np.ndarray]:
    target=tensor_rows(rows,device); errors=[]
    with torch.no_grad():
        _,embedding=model(target)
        for offset in (0.0,0.25,0.5,0.75):
            masked,mask=_mask(target,0,offset); reconstruction,_=model(masked); error=(((reconstruction-target)**2*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1.0)); errors.append(error.cpu().numpy())
    return np.mean(errors,axis=0),embedding.cpu().numpy()


def _combined_scores(reference_rows: np.ndarray, query_rows: np.ndarray, model: MaskedAutoencoder, device: str) -> tuple[np.ndarray,np.ndarray]:
    reference_error,reference_embedding=_model_outputs(model,reference_rows,device); query_error,query_embedding=_model_outputs(model,query_rows,device)
    neighbors=NearestNeighbors(n_neighbors=max(1,min(3,len(reference_embedding)))).fit(reference_embedding)
    reference_distance=neighbors.kneighbors(reference_embedding)[0][:,-1]; query_distance=neighbors.kneighbors(query_embedding)[0].mean(axis=1)
    def scale(reference:np.ndarray,query:np.ndarray)->np.ndarray:
        center=np.median(reference); spread=1.4826*np.median(np.abs(reference-center))+1e-8; return np.maximum(0.0,(query-center)/spread)
    return 0.5*scale(reference_error,query_error)+0.5*scale(reference_distance,query_distance),query_embedding


class MaskedPatchContextError:
    name="masked_patch_context_error"; minimum_train=20

    def __init__(self,epochs:int=12,seeds:tuple[int,...]|None=None,device:str="cpu")->None:
        self.epochs=int(epochs);self.seeds=seeds;self.device=device

    def score(self,bundle:DatasetBundle,seed:int,artifact_dir:Path)->ScoreResult:
        if len(bundle.train_normal)<self.minimum_train:return abstained_result(self.name,bundle,self.minimum_train)
        train=prepare_rows(bundle.train_normal);test=prepare_rows(bundle.test_values);seeds=self.seeds or (int(seed),SECOND_SEED);train_scores=[];test_scores=[];train_embeddings=[];test_embeddings=[]
        for current_seed in seeds:
            oof=np.empty(len(train));oof_embedding=np.empty((len(train),32),dtype=np.float32)
            for fold,(fit,query) in enumerate(crossfit_plan(len(train),seed=current_seed)):
                model=_fit_model(train[fit],current_seed+fold,self.epochs,self.device);oof[query],oof_embedding[query]=_combined_scores(train[fit],train[query],model,self.device)
            model=_fit_model(train,current_seed,self.epochs,self.device);test_score,test_embedding=_combined_scores(train,test,model,self.device)
            train_scores.append(oof);test_scores.append(test_score);train_embeddings.append(oof_embedding);test_embeddings.append(test_embedding)
        artifacts={"train_embeddings":np.mean(train_embeddings,axis=0).astype(np.float32),"test_embeddings":np.mean(test_embeddings,axis=0).astype(np.float32),"train_scores_by_seed":np.asarray(train_scores),"test_scores_by_seed":np.asarray(test_scores)}
        return ScoreResult(self.name,np.median(train_scores,axis=0),np.median(test_scores,axis=0),False,None,{"seeds":list(seeds),"epochs":self.epochs,"mask_ratio":0.25,"embedding_dimension":32,"device":self.device,"cross_fitted":True,"self_reference_violations":0,"contains_test_labels":False},artifacts=artifacts)

