from jax import vmap, lax
import jax.numpy as jnp
import equinox as eqx

from loss import LLT_loss, mse_loss

def compute_loss_LLT(model, X, y):
    '''Graph was made out of lhs A.
       Positions in `X`:
         X[0] - nodes of the graph.
         X[1] - edges of the graph.
         X[2] - receivers of the graph.
         X[3] - senders of the graph.
         X[4] - indices of bi-directional edges in the graph.
         X[5] - solution of linear system x.
         X[6] - rhs b.
     '''
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(X[0], X[1], X[2], X[3], X[4])
    loss = vmap(LLT_loss, in_axes=(0, 0, 0), out_axes=(0))(L, X[5], X[6])
    return jnp.sum(loss)

def compute_loss_mse(model, X, y):
    '''Graph was made out of lhs A.
       Positions in `X`:
         X[0] - nodes of the graph.
         X[1] - edges of the graph.
         X[2] - receivers of the graph.
         X[3] - senders of the graph.
         X[4] - indices of bi-directional edges in the graph.
         X[5] - solution of linear system x.
         X[6] - rhs b.
         X[7] - lhs A.
     '''
    L = vmap(model, in_axes=(0, 0, 0, 0, 0), out_axes=(0))(X[0], X[1], X[2], X[3], X[4])#(nodes, edges, X[2], X[3], X[4])#(X[0], X[1], X[2], X[3], X[4])
    loss = vmap(mse_loss, in_axes=(0, 0), out_axes=(0))(L, X[7])#(L, solution, b)#(L, X[5], X[6])
    return jnp.mean(loss)

def train(model, data, train_config, compute_loss):
    assert isinstance(train_config, dict)
    assert isinstance(data, tuple)
    assert len(data) == 4
    
    X_train, X_test, y_train, y_test = data
    assert isinstance(X_train, tuple)
    assert isinstance(X_test, tuple)
    
    optim = train_config['optimizer'](train_config['lr'], **train_config['optim_params'])
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    
    compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)
    
    def make_step(model, X, y, opt_state):
        loss, grads = compute_loss_and_grads(model, X, y)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    
    def make_val_step(model, X, y):
        loss = compute_loss(model, X, y)
        return loss
    
    def train_body(carry, x):
        model, opt_state = carry
        loss_train, model, opt_state = make_step(model, X_train, y_train, opt_state)
        loss_test = make_val_step(model, X_test, y_test)
        carry = (model, opt_state)
        return carry, [loss_train, loss_test]
    
    carry_init = (model, opt_state)
    (model, _), losses = lax.scan(train_body, carry_init, None, length=train_config['epoch_num'])
    return model, losses









    
#     def train_body(model, X_train, X_test, y_train, y_test, opt_state):
#         loss_train, model, opt_state = make_step(model, X_train, y_train, opt_state)
#         loss_test = make_val_step(model, X_test, y_test)
#         return model, opt_state, loss_train, loss_test
#     loss_ls = []
#     for ep in range(train_config['epoch_num']):
#         model, opt_state, loss_train, loss_test = train_body(model, X_train, X_test, y_train, y_test, opt_state)
#         print(f'Epoch: {ep}, train loss: {loss_train}, test loss:{loss_test}')
#         loss_ls.append([loss_train, loss_test])
#     return model, jnp.stack(jnp.asarray(loss_ls))