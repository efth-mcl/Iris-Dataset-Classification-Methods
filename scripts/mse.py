from Irisdataset import *

# Methods
def normal_eq(X,Y):
    W = np.linalg.inv(np.dot(X.T, X))
    W = np.dot(np.dot(W, X.T), Y)
    return W


iris = Irisdataset(150)

def_tr_index = lambda x,n:[False if j==x else True for j in range(n)]
error = 0
S_Examples = iris.TrainData
N = S_Examples.shape[0]

S_Data = S_Examples[:,:-1]
S_Data = np.concatenate((np.ones((N,1)),S_Data),axis=1)
S_Hot = np.zeros((N,3))
S_Hot[range(N),S_Examples[:,-1].astype(int)] = 1
for i in range(N):
    tr_index = np.array(def_tr_index(i,N))

    tr_data = S_Data[tr_index]
    tr_hot = np.array([S_Hot[tr_index] == 1]).reshape(-1,3)
    
    ts_data = S_Data[i].reshape(1,-1)
    ts_hot = np.array([S_Hot[i] == 1]).reshape(-1,3)
    W = normal_eq(tr_data,tr_hot)
    W[0,:]+=0
    ts_pr = np.dot(ts_data,W)
    cl_p = np.sum(ts_pr.argmax()!=ts_hot.argmax())
    error += cl_p
    
print('Test error: {0:.2f}'.format(error/N))

# Plot-Data

f0 = 3 # first feature 
f1 = 1 # second feature
f1+=1
f0+=1
c1_tr = S_Data[np.where(S_Hot.argmax(1) == 0)]
c2_tr = S_Data[np.where(S_Hot.argmax(1) == 1)]
c3_tr = S_Data[np.where(S_Hot.argmax(1) == 2)]

fig, ax = plt.subplots(ncols=2, figsize=(8, 4), dpi= 80, facecolor='w', edgecolor='k' )
fig.subplots_adjust(
    left  = 0.08,
    right = 0.97,
    wspace=0.4,
    hspace = 0.2,
    top=0.90
)

ax[0].plot(
        c1_tr[:,f0],c1_tr[:,f1],'o',
        c2_tr[:,f0],c2_tr[:,f1],'o',
        c3_tr[:,f0],c3_tr[:,f1],'o'
)

ax[0].set_title('true labels')
ax[0].set_xlabel(iris.feature_names[f0-1])
ax[0].set_ylabel(iris.feature_names[f1-1])

pr_d = S_Data.dot(W) 
pl1 = S_Data[np.where(pr_d.argmax(1)== 0)[0]][:,[f0,f1]]
pl2 = S_Data[np.where(pr_d.argmax(1) == 1)[0]][:,[f0,f1]]
pl3 = S_Data[np.where(pr_d.argmax(1) == 2)[0]][:,[f0,f1]]

sc1, sc2, sc3 = ax[1].plot(
                        pl1[:,0],pl1[:,1],'o',
                        pl2[:,0],pl2[:,1],'o',
                        pl3[:,0],pl3[:,1],'o'
                    )
ax[1].set_title('predicted labels')
ax[1].set_xlabel(iris.feature_names[f0-1])
ax[1].set_ylabel(iris.feature_names[f1-1])

fig.legend((sc1, sc2, sc3),
           tuple(iris.set_names),
           'upper center',
           fontsize=16,
           title='classes',
           bbox_to_anchor=[0.55, 1.5]
);

plt.show()
