import numpy as np


def Predict(X,W_L_1,W_L_2,L_1_b,L_2_b,Y):
	
	for i in range(100):
		l = np.array([X[i]])
		L_1_Z = forwar_Pass(l.T, W_L_1, L_1_b)
		L_1_A = sigmoid(L_1_Z)
		L_2_Z = forwar_Pass(L_1_A, W_L_2, L_2_b)
		activation = sigmoid(L_2_Z)
		label = 0 if activation <0.5 else 1
		print("activation={}; predicted_label={}, true_label={}".format(activation, label, Y[i]))

def tanh(Z):
	return 
def sigmoid(Z):
	return (1 / (1 + np.exp(-Z)))
def relu():
	return
def sigmoid_derivativa(X):
	return sigmoid(X) * (1 - sigmoid(X))
def forwar_Pass(X,W,b):
	return np.dot(W,X) + b
def derivative_w_r_t_to_weights(dz, A, m):
	return (np.dot(dz,A) / m )
def Getting_Data_Set_Ready():
	b_1 = b_2 = 1
	split = 150
	data_set = np.genfromtxt('data.csv',delimiter = ' ')
	np.random.shuffle(data_set)
	
	train,test = data_set[:split,:],data_set[split:,:]

	W_1 = np.random.rand(100,324)
	W_2 = np.random.rand(1,100)


	X,Y = train[:,:-1],train[:,-1]
	X = X.T 
	Y = np.array([Y])

	print("shape of X ",X.shape)
	print("Shape of Y ",Y.shape)
	print("shape of Layer 1 weights ",W_1.shape)
	print("Shape of Layer 2 weights ",W_2.shape)
	return X,Y,test,W_1,W_2,b_1,b_2

def Neural_Network():
	X,Y,test,W_L_1,W_L_2,L_1_b,L_2_b = Getting_Data_Set_Ready()
	print(W_L_2.shape)
	alpha = 0.001
	m = np.prod(X.shape)

	for i in range(1000):
		L_1_Z = forwar_Pass(X, W_L_1, L_1_b)
		L_1_A = sigmoid(L_1_Z)
		L_2_Z = forwar_Pass(L_1_A, W_L_2, L_2_b)
		L_2_A = sigmoid(L_2_Z)

		#backward Pass
		L_2_dz = L_2_A - Y

		L_2_dw = derivative_w_r_t_to_weights(L_2_dz, L_1_A.T,m)
		L_2_db = np.sum(L_2_dz, axis = 1, keepdims = True)
		L_1_dz = np.dot(W_L_2.T,L_2_dz) * sigmoid_derivativa(L_1_Z)
		L_1_dw = derivative_w_r_t_to_weights(L_1_dz, X.T, m)
		L_1_db = (np.sum(L_1_dz, axis = 1, keepdims = True))
		error_1 = np.sum(L_1_dz**2)
		error_2 = np.sum(L_2_dz**2)
		print("layer 1 error is ",error_1)
		print("layer 2 error is ", error_2)
		W_L_1 -=  0.01* W_L_1
		W_L_2 -=  0.01* W_L_2

		L_1_b -= 0.01 * L_1_b
		L_2_b -= 0.01 * L_2_b
		


	New_X, New_Y = test[:,:-1],test[:,-1]
	New_Y = np.array([New_Y])
	print("W_1",W_L_1.shape)
	print("X ",New_X.shape)
	print("Y ",New_Y.shape)
	Predict(New_X,W_L_1,W_L_2,L_1_b,L_2_b,New_Y.T)

	
Neural_Network()