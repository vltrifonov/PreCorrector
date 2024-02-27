import jax.numpy as jnp

from utils import JaxTrainer

class TrainerLLT(JaxTrainer):
    def __init__(self, X_train, X_test, y_train, y_test, training_params, loss_function):
        super(TrainerLLT, self).__init__(X_train, X_test, y_train, y_test, training_params)
        self.loss_func = loss_function
        return
    
    def compute_loss(self, model, *args):
        '''args[0] - batched_X_train; args[1] - batched_y_train
            args[0][1]- graph (made out of A)
            args[0][1] - x
            args[0][2] - b
            args[0][3] - bi-directional edges indices'''
        
        L = model(args[0][0], args[0][3])
        loss = self.loss_func(L, args[0][1], args[0][b])
        return loss
    
    def filter_batch(self, jarr_ind, not_jarray_ind, **kwargs):
        self.batched_X_train = [i for i in range(len(self.X_train))]
        for i in jarr_ind:
            self.batched_X_train[i] = self.X_train[i][train_batch_indices, ...]
        for i in not_jarrat_ind:
            tuple_graph = self.X_train[i]
            
            self.batched_X_train[i] = batched_graph
        pass
    
    def filter_batch_array(self, arr_indx, train_batch_indices, test_batch_indices, **kwargs):
        self.batched_X_train = []
        for i in arr_indx:
            self.batched_X_train.append(self.X_train[i][train_batch_indices, ...])
        self.batched_X_test = self.X_test[None, ...]
        return
    
    def filter_batch_jraph_tuple(self, ):
        pass