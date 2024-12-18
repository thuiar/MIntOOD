# import numpy as np
# from sklearn.covariance import EmpiricalCovariance
# from numpy.linalg import norm, pinv
# from scipy.special import logsumexp

# def func(args, inputs):
    
#     w = inputs['w']
#     w = w.data.cpu().numpy()
#     b = inputs['b']
#     b = b.data.cpu().numpy()
    
#     train_feats = inputs['train_feats']
#     train_logits = train_feats @ w.T + b

#     u = -np.matmul(pinv(w), b)
            
#     ec = EmpiricalCovariance(assume_centered=True)
#     ec.fit(train_feats - u)
#     eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
#     NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[args.num_labels:]]).T)
#     # 主子空间p的正交补

#     vlogit_ind_train = norm(np.matmul(train_feats - u, NS), axis=-1)
#     alpha = train_logits.max(axis=-1).mean() / vlogit_ind_train.mean()

#     features = inputs['y_feat']
#     logit = features @ w.T + b
#     vlogit = norm(np.matmul(features - u, NS), axis=-1) * alpha
#     energy = logsumexp(logit, axis=-1)
#     scores = -vlogit + energy

#     return scores
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from numpy.linalg import norm, pinv
from scipy.special import logsumexp

def func(args, inputs):
    
    w = inputs['w']
    w = w.data.cpu().numpy()    

    b = inputs['b']
    b = b.data.cpu().numpy()
    
    train_feats = inputs['train_feats']
    print(w.shape)
    if args.method == 'mmco':
        w /= np.linalg.norm(w, ord=2, axis=1, keepdims=True)
        train_feats /= np.linalg.norm(train_feats, ord=2, axis=1, keepdims=True)

    train_logits = (train_feats * args.scale) @ w.T + b #

    u = -np.matmul(pinv(w), b)
            
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(train_feats - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[args.num_labels:]]).T)
    # 主子空间p的正交补

    if args.method == 'mmco':
        vlogit_ind_train = norm(np.matmul((train_feats - u) * args.scale, NS), axis=-1)
    else:
        vlogit_ind_train = norm(np.matmul(train_feats - u, NS), axis=-1)

    alpha = train_logits.max(axis=-1).mean() / vlogit_ind_train.mean()

    features = inputs['y_feat']
    if args.method == 'mmco':
        logit = (features * args.scale) @ w.T + b #
        vlogit = norm(np.matmul((features - u) * args.scale, NS), axis=-1) * alpha
    else:
        logit = features @ w.T + b #
        vlogit = norm(np.matmul(features - u, NS), axis=-1) * alpha
    energy = logsumexp(logit, axis=-1)
    scores = -vlogit + energy

    return scores