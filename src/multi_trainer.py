# import sys
# sys.path.append('..')
import torch
import numpy as np
from utils.earlystopping import EarlyStopping
from utils.meter import Meter
import time
class Multi_Trainer(object):
    def __init__(self, device, task1, task2, data_args1, data_args2, model_path, 
                 n_epochs=1000, patience=20, inter_print=20):
        print("initializing trainer...")
        self.device = device
        self.tasks = [task1, task2]
        self.model_path = model_path
        self.metrics = [data_args1['metrics'], data_args2['metrics']]
        self.norm = [data_args1['norm'], data_args2['norm']]
        if self.norm[0]:
            self.data_mean1 = torch.tensor(data_args1['mean']).to(self.device)
            self.data_std1 = torch.tensor(data_args1['std']).to(self.device)
        if self.norm[1]:
            self.data_mean2 = torch.tensor(data_args2['mean']).to(self.device)
            self.data_std2 = torch.tensor(data_args2['std']).to(self.device)
        self.loss_fn = [data_args1['loss_fn'], data_args2['loss_fn']]
        self.n_epochs = n_epochs
        self.patience=patience
        self.inter_print = inter_print
    
    def _prepare_batch_data(self,batch_data):
        smiless, inputs, labels, masks = batch_data
        inputs = inputs.to(self.device) #https://docs.dgl.ai/en/0.8.x/guide/graph-gpu.html
        inputs.ndata['h'] = inputs.ndata['h'].to(self.device)
        labels = labels.to(self.device)
        masks = masks.to(self.device)
        return smiless, inputs, labels, masks
    
    def _train_epoch(self, model, train_loader1, train_loader2, loss_fn, optimizer):
        model.train()
        loss_list = []
        for i, all_batch_data in enumerate(zip(train_loader1,train_loader2)):
            losses = []
            for task_id, batch_data in enumerate(all_batch_data): 
                _, inputs, labels, masks = self._prepare_batch_data(batch_data)
                _, predictions = model(inputs, task_id)
                if self.norm[task_id]:
                    if task_id == 0:
                        labels = (labels - self.data_mean1)/self.data_std1
                    else:
                        labels = (labels - self.data_mean2)/self.data_std2
                losses.append((loss_fn[task_id](predictions, labels)*(masks!=0).float()).mean())
            loss = losses[0] + losses[1]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        return np.mean(loss_list)
    
    def _eval(self, model, data_loader, task_id):
        model.eval()
        meter = Meter([self.tasks[task_id]])
        for i, batch_data in enumerate(data_loader):
            smiless, inputs, labels, masks = self._prepare_batch_data(batch_data)
            _, predictions = model(inputs, task_id)
            if self.norm[task_id]:
                if task_id == 0:
                    predictions = predictions * self.data_std1 + self.data_mean1
                else:
                    predictions = predictions * self.data_std2 + self.data_mean2
            meter.update(predictions, labels, masks)
        eval_results_dict = meter.compute_metric(self.metrics[task_id])
        return eval_results_dict
    
    def _train(self, model, train_loader1, val_loader1, train_loader2, val_loader2, loss_fn, optimizer, stopper1, stopper2):
        first_stopped = None 
        for epoch in range(self.n_epochs):
            loss = self._train_epoch(model, train_loader1, train_loader2, loss_fn,
                                                 optimizer)
            if epoch % self.inter_print == 0:
                print(f"[{epoch}] training loss:{loss}")
            for task_id, val_loader, stopper in [(0, val_loader1, stopper1), (1, val_loader2, stopper2)]:
                val_results_dict = self._eval(model, val_loader, task_id)
                early_stop = stopper.step(val_results_dict[self.metrics[task_id][0]]['mean'],
                                         model, epoch)
                if epoch % self.inter_print == 0:
                    print(f"{self.tasks[task_id]}")
                    for metric in self.metrics[task_id]:
                        print(f"val {metric}:{val_results_dict[metric]['mean']}")
                if early_stop and first_stopped is None:
                    first_stopped = task_id
                if early_stop and first_stopped != task_id:
                    return task_id #Second task done now 
    
    def fit(self, model, train_loader1, val_loader1, test_loader1, train_loader2, val_loader2, test_loader2):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                            model.parameters()), 
                                     lr=1e-4, weight_decay=1e-5)
        stopper1 = EarlyStopping(self.model_path,self.tasks, patience=self.patience) #Still need to include both task names to save checkpoints to same location
        stopper2 = EarlyStopping(self.model_path,self.tasks, patience=self.patience)
        
        stopper_task_id = self._train(model,train_loader1,val_loader1,train_loader2,val_loader2,
                            self.loss_fn,optimizer,stopper1, stopper2)
        if stopper_task_id == 0:
            stopper1.load_checkpoint(model)
        else:
            stopper2.load_checkpoint(model)
        test_results1_dict = self._eval(model, test_loader1, 0)
        test_results2_dict = self._eval(model, test_loader2, 1)
        for metric in self.metrics[0]:
            print(f"test {metric}:{test_results1_dict[metric]['mean']}")
        for metric in self.metrics[1]:
            print(f"test {metric}:{test_results2_dict[metric]['mean']}")
        return model, test_results1_dict, test_results2_dict