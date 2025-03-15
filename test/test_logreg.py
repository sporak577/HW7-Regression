"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression.logreg import LogisticRegressor



@pytest.fixture
def setup_model():
	"""
	setting up a small dataset and LogisticRegressor instance for testing.
	"""

	X = np.array([
	[1, 2.0, 3.0], #bias term is included
	[1, 1.0, 1.5],
	[1, 2.5, 2.0]
	])

	y = np.array([1, 0, 1]) #true labels
	W = np.array([0.2, -0.4, 0.6]) #initial weights 

	#initialize LogisticRegressor 
	model = LogisticRegressor(num_feats=2)
	model.W = W #manually set weights for testing

	return model, X, y

def test_prediction(setup_model):
	"""
	test if make_prediction correcly computes probabilities between 0 and 1. 
	"""
	model, X, _ = setup_model
	y_pred = model.make_prediction(X)
	
	assert np.all(y_pred >= 0) and np.all(y_pred <= 1) #probs should be between 0 and 1, as this is logistic regression. 


def test_loss_function(setup_model):
	"""
	test if loss_function correctly computes binary cross-entropy loss. 
	"""
	model, X, y_true = setup_model
	y_pred = model.make_prediction(X)
	loss = model.loss_function(y_true, y_pred)

	epsilon = 1e-8 #to avoid log0
	expected_loss = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

	assert np.isclose(loss, expected_loss, atol=1e-5)

def test_gradient(setup_model):
	"""
	test if the calculate_gradient correctly computes the gradient
	"""
	model, X, y_true = setup_model
	y_pred = model.make_prediction(X)
	computed_grad = model.calculate_gradient(y_true, X)

	#manually computed expected gradient
	expected_grad = np.dot(X.T, (y_pred - y_true)) / X.shape[0]

	np.testing.assert_array_almost_equal(computed_grad, expected_grad, decimal=5, err_msg="Gradient calculation is incorrect.")
	

def test_training(setup_model):
	"""
	test if train_model updates weights correctly after one step. 
	"""
	model, X, y_true = setup_model
	initial_W = model.W.copy()

	#train for one iteration
	model.train_model(X, y_true, X, y_true) #using same data for validation

	assert not np.array_equal(initial_W, model.W) #weights should be uppdated after training