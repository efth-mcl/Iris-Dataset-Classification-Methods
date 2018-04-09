from Irisdataset import *

# Methods
Hot = lambda x:[1 if i==x else -1 for i in range(3)]

 
learning_rate = 0.7
total_test_error = 0

f0 = 3
f1 = 1
for cnt in range(5):
    iris = Irisdataset(120)  # 120 train data
    train_data = iris.TrainData[:,[f0,f1]]
    train_hot = -1*np.ones((120,3))
    train_hot[range(120),iris.TrainData[:,-1].astype(int)] = 1
    
    test_data = iris.TestData[:,[f0,f1]]
    test_hot = -1*np.ones((30,3))
    test_hot[range(30),iris.TestData[:,-1].astype(int)] = 1
    
    W = np.random.rand(2, 3) # Weights
    B = np.random.rand(1, 3) # Bias
    
    # Training
    # update W, B 200 times or Epochs = 200
    for i in range(200): 
        tr_error = 0
        for tr_d,tr_hot in zip(train_data,train_hot):
            # forward pass
            tr_d = tr_d.reshape(1, 2)
            neurons_outs = np.dot(tr_d, W)+B
            
            # outputs indexis with wrong prediction class
            tr_error+=np.where(neurons_outs*tr_hot<0)[0].size>0
            
            # backward pass
            neurons_outs[np.where(neurons_outs*tr_hot>0)] = 0
            neurons_outs = np.sign(neurons_outs)
            
            # Deltas
            DW = -learning_rate*np.dot(tr_d.T,neurons_outs)
            DB = -learning_rate*neurons_outs
            
            # Update
            W += DW
            B += DB
            
    test_neurons_outs = np.dot(test_data, W)+B
    test_prediction = test_neurons_outs.argmax(1)
    ts_error = sum(test_hot.argmax(1) != test_prediction)
    total_test_error += ts_error
    
    print('Round {0:d}: Test error={1:.2f}'.format(cnt+1, ts_error/30))

print('\nAverage Test error={0:.2f}'.format(total_test_error/(5*30)))

# Plot-Data

# True labels
c1_tr_train = train_data[np.where(iris.TrainData[:,4] == 0)]
c2_tr_train = train_data[np.where(iris.TrainData[:,4] == 1)]
c3_tr_train = train_data[np.where(iris.TrainData[:,4] == 2)]

c1_tr_test = test_data[np.where(iris.TestData[:,4] == 0)]
c2_tr_test = test_data[np.where(iris.TestData[:,4] == 1)]
c3_tr_test = test_data[np.where(iris.TestData[:,4] == 2)]

# Predicted labels for test examples
c1_pr_test = test_data[test_prediction == 0]
c2_pr_test = test_data[test_prediction == 1]
c3_pr_test = test_data[test_prediction == 2]

fig, ax = plt.subplots(figsize=(8, 8), dpi= 80, facecolor='w', edgecolor='k' )
fig.subplots_adjust(
    left  = 0.10,
    right = 0.90,
    bottom = 0.10,
    wspace=0.4,
    hspace = 0.2,
    top=0.90
)

c1_c = 'royalblue'
c2_c = 'darkorange'
c3_c = 'limegreen'

ax.plot(c1_tr_train[:,0],c1_tr_train[:,1],'o',color=c1_c)
ax.plot(c2_tr_train[:,0],c2_tr_train[:,1],'o',color=c2_c)
ax.plot(c3_tr_train[:,0],c3_tr_train[:,1],'o',color=c3_c)

ms_c1 = dict(color=c1_c, marker='o')
ms_c2 = dict(color=c2_c, marker='o')
ms_c3 = dict(color=c3_c, marker='o')

ax.plot(c1_tr_test[:,0],c1_tr_test[:,1],'o', **ms_c1, markersize=15, fillstyle='none')
ax.plot(c2_tr_test[:,0],c2_tr_test[:,1],'o', **ms_c2, markersize=15, fillstyle='none')
ax.plot(c3_tr_test[:,0],c3_tr_test[:,1],'o', **ms_c3, markersize=15, fillstyle='none')


ax.plot(c1_pr_test[:,0],c1_pr_test[:,1],'o', **ms_c1)
ax.plot(c2_pr_test[:,0],c2_pr_test[:,1],'o', **ms_c2)
ax.plot(c3_pr_test[:,0],c3_pr_test[:,1],'o', **ms_c3)

ax.set_xlabel(iris.feature_names[f0], fontsize=20)
ax.set_ylabel(iris.feature_names[f1], fontsize=20)

plt.show()
