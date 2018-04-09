from Irisdataset import *
import matplotlib.mlab as mlab
# Methods
Normal_dist = lambda x,avg, std: 1/np.sqrt(2*np.pi*std**2)*np.exp(-(x-avg)**2/(2*std**2))

ND_mahalanobis = lambda x,mean,S: np.dot(np.dot((x-mean).T,np.linalg.inv(S)),(x-mean))**(1/2)

def ND_normal_dist(x,mean,S):
    m_dist = ND_mahalanobis(x, mean, S)
    constant = 1/((2*np.pi)**(4/2)*np.linalg.det(S)**(1/2))
    return constant*np.exp(-1/2*np.square(m_dist))


iris = Irisdataset(150) # all dataset

error_h1=0 # hypothesis_1 error
error_h2=0 # hypothesis_2 error

def_tr_index = lambda x:[False if j==x else True for j in range(150)]
for i in range(150):
    
    tr_index = np.array(def_tr_index(i))
    tr_data = iris.TrainData[tr_index]
    # get i_th record for validation
    vl_data = iris.TrainData[i]
    
    
    # Estimate maximum likelihood per feature
    # AVG - STD
    # Setosa
    tr_setosa = tr_data[np.where(tr_data[:,4] == 0)][:,:4]
    tr_setosa_avg_per_feature = np.mean(tr_setosa,axis=0)
    tr_setosa_std_per_feature = np.std(tr_setosa, axis=0)

    # Versicolor
    tr_versicolor = tr_data[np.where(tr_data[:,4] == 1)][:,:4]
    tr_versicolor_avg_per_feature = np.mean(tr_versicolor, axis=0)
    tr_versicolor_std_per_feature = np.std(tr_versicolor, axis=0)

    # Virginica
    tr_virginica = tr_data[np.where(tr_data[:,4] == 2)][:,:4]
    tr_virginica_avg_per_feature = np.mean(tr_virginica, axis=0)
    tr_virginica_std_per_feature = np.std(tr_virginica, axis=0)
    
    
    # hypothesis 1 START
    # Test Model
    # class confidence per feature
    h1_conf_setosa = Normal_dist(vl_data[:-1], tr_setosa_avg_per_feature, tr_setosa_std_per_feature)
    h1_conf_versicolor = Normal_dist(vl_data[:-1], tr_versicolor_avg_per_feature, tr_versicolor_std_per_feature)
    h1_conf_virginica = Normal_dist(vl_data[:-1], tr_virginica_avg_per_feature, tr_virginica_std_per_feature)
    
    # prediction_class
    arg_max_class_conf_per_feature = np.argmax([
            h1_conf_setosa,
            h1_conf_versicolor,
            h1_conf_virginica      
        ], axis=0)
    counts = np.bincount(arg_max_class_conf_per_feature)
    h1_prediction_class = np.argmax(counts)
    
    # add 1 if true_class != prediction_class
    error_h1 += h1_prediction_class != vl_data[-1]
    # hypothesis 1 END
    
    # hypothesis 2 START
    # calculate converance by class
    tr_setosa_cov = np.cov((tr_setosa).T)
    tr_versicolor_cov = np.cov((tr_versicolor).T)
    tr_virginica_cov = np.cov((tr_virginica).T)
    
    # class confidence 
    h2_conf_setosa = ND_normal_dist(vl_data[:-1].reshape(4,1),tr_setosa_avg_per_feature.reshape(4,1),tr_setosa_cov)
    h2_conf_versicolor = ND_normal_dist(vl_data[:-1].reshape(4,1),tr_versicolor_avg_per_feature.reshape(4,1),tr_versicolor_cov)
    h2_conf_virginica = ND_normal_dist(vl_data[:-1].reshape(4,1),tr_virginica_avg_per_feature.reshape(4,1),tr_virginica_cov)
    
    h2_prediction_class = np.argmax([h2_conf_setosa, h2_conf_versicolor, h2_conf_virginica])
    
    # add 1 if true_class != prediction_class 
    error_h2 += h2_prediction_class != vl_data[-1]
    # hypothesis 2 END
    
print(error_h1/150)
print(error_h2/150)

# Plot-Data

f0 = 0
f1 = 1
min_x = min(iris.TrainData[:,f0])-0.1
max_x = max(iris.TrainData[:,f0])+0.1
min_y = min(iris.TrainData[:,f1])-0.1
max_y = max(iris.TrainData[:,f1])+0.1
cmap = plt.cm.magma
y = np.linspace(min_x, max_x)
x = np.linspace(min_y, max_y)

X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots(2,2, figsize=(14, 14), dpi= 80, facecolor='w', edgecolor='k' )

for i in range(2):
    sc1, sc2, sc3 = ax[i,0].plot(
        tr_setosa[:,f1],tr_setosa[:,f0],'o',
        tr_versicolor[:,f1],tr_versicolor[:,f0],'o',
        tr_virginica[:,f1],tr_virginica[:,f0],'o'
    )

fig.legend((sc1, sc2, sc3),
           tuple(iris.set_names),
           'upper center',
           fontsize=16,
           title='classes',
          );

fig.text(0.5, 0.04, Irisdataset.feature_names[f1], ha='center',fontsize=22)
fig.text(0.04, 0.5, Irisdataset.feature_names[f0], va='center', rotation='vertical',fontsize=22)

left  = 0.125
right = 1
bottom = 0.1
top = 0.85
wspace = 0.2
hspace = 0.3

fig.subplots_adjust(
    left  = left,
    right = right,
    bottom =bottom,
    top = top,
    wspace = wspace,
    hspace = hspace
)

# hypothesis 1
var_set = tr_setosa_std_per_feature[[f0,f1]]
avg_set = tr_setosa_avg_per_feature[[f0,f1]]
Z_set = mlab.bivariate_normal(X, Y, var_set[1], var_set[0], avg_set[1], avg_set[0])

var_ver = tr_versicolor_std_per_feature[[f0,f1]]
avg_ver = tr_versicolor_avg_per_feature[[f0,f1]]
Z_ver = mlab.bivariate_normal(X, Y, var_ver[1], var_ver[0], avg_ver[1], avg_ver[0])

var_vir = tr_virginica_std_per_feature[[f0,f1]]
avg_vir = tr_virginica_avg_per_feature[[f0,f1]]
Z_vir = mlab.bivariate_normal(X, Y, var_vir[1], var_vir[0], avg_vir[1], avg_vir[0])
Z = Z_set+Z_ver+Z_vir

ax[0,0].contour(X, Y, Z_set,cmap = plt.cm.winter_r)
ax[0,0].contour(X, Y, Z_ver,cmap = plt.cm.autumn)
ax[0,0].contour(X, Y, Z_vir,cmap = plt.cm.summer_r)
ax[0,0].set_title('Naive Base Hypothesis_1 classifiers')

ax[0,1].axis('off')
ax[0,1].imshow(Z, interpolation='bilinear', cmap=cmap,
                origin='lower', extent=[min_x, max_x, min_y, max_y],
                )

ax[0,1].set_title('Hypothesis_1 Classifiers magnitude')

# hypothesis 2
cov_set = np.cov((tr_setosa[:,[f0,f1]]).T).flatten()
cov_set[[0,-1]] = cov_set[[0,-1]]**(1/2)
Z_set_cov = mlab.bivariate_normal(X, Y, cov_set[-1], cov_set[0], avg_set[1], avg_set[0],cov_set[1])

cov_ver = np.cov((tr_versicolor[:,[f0,f1]]).T).flatten()**(1/2)
cov_ver[[0,-1]] = cov_ver[[0,-1]]**(1/2)
Z_ver_cov = mlab.bivariate_normal(X, Y, cov_ver[-1], cov_ver[0], avg_ver[1], avg_ver[0],cov_ver[1])

cov_vir = np.cov((tr_virginica[:,[f0,f1]]).T).flatten()**(1/2)
cov_vir[[0,-1]] = cov_vir[[0,-1]]**(1/2)
Z_vir_cov = mlab.bivariate_normal(X, Y, cov_vir[-1], cov_vir[0], avg_vir[1], avg_vir[0],cov_vir[1])

ax[1,0].contour(X, Y, Z_set_cov,cmap = plt.cm.winter_r)
ax[1,0].contour(X, Y, Z_ver_cov,cmap = plt.cm.autumn)
ax[1,0].contour(X, Y, Z_vir_cov,cmap = plt.cm.summer_r)


Z_cov = Z_set_cov+Z_ver_cov+Z_vir_cov
ax[1,1].axis('off')
ax[1,1].imshow(Z_cov, interpolation='bilinear', cmap=cmap,
                origin='lower', extent=[min_x, max_x, min_y, max_y],
                )

ax[1,0].set_title('Naive Base Hypothesis_2 classifiers')
ax[1,1].set_title('Hypothesis_2 Classifiers magnitude')
plt.show()
