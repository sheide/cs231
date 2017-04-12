import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
    sum = 0.0
    x = X[i]
    for k in range(W.shape[1]):
      w = W[:,k]
      f_k = x.dot(w)
      sum += np.exp(f_k)
    f_y_i = x.dot(W[:,y[i]])
    loss += -np.log(np.exp(f_y_i)/sum)
    for j in range(W.shape[1]):
      w = W[:,j]
      f_ij = np.dot(x,w)
      p = np.exp(f_ij)/sum
      dW[:,j] += (p - (j==y[i]))*x
  
  loss /= X.shape[0]    
  loss += reg*0.5*np.sum(W*W)

  dW /= X.shape[0]
  dW += reg*W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = np.dot(X,W)
  f -= np.max(f)
  e_f = np.exp(f)
  e_f_y = e_f[np.arange(y.shape[0]), y]
  e_f_sum = np.sum(e_f,1)
  p = e_f / e_f_sum[:,None]
  p[np.arange(num_train),y] -= 1
  dW = (np.dot(p.T,X)/num_train).T + reg*W
  loss = np.mean(-np.log(e_f_y/e_f_sum)) + reg*0.5*np.sum(W*W)
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

