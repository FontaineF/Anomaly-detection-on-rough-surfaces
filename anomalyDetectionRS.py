# Import libraries
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import imageio

# Get CSV files list from a folder
path = "data/*"
csv_files = glob.glob(path)

# Read each CSV file into DataFrame
# This creates a list of dataframes
df_list = (pd.read_csv(file) for file in csv_files)

# Concatenate all DataFrames
big_df = pd.concat(df_list, ignore_index=False)
df = big_df.to_numpy()
length = len(df[0])
x = df.reshape(-1,length,length,1)

# Create train and test arrays
def normalize(X):
    x = np.copy(X)
    for i in range(len(x)):
        x[i]=(x[i]-x[i].min())/(x[i].max()-x[i].min())
    return x

x_norm = normalize(x)

def gaussian_blur(img, kernel_size=11, sigma=5):
    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
                                  padding='SAME', data_format='NHWC')

# Take random parts of the original image to create new aarays for our network
# s: scale, t: length of the new arrays, n: number of new arrays per original image
def random_cut(X,s=1,t=48,n=1):
    if s == 1:
        x = X.copy()
    else:
        x = gaussian_blur(X.astype('float32'))
    i_max = min(x.shape[1],x.shape[2])-t*s
    indices = np.random.randint(10,i_max-10,(len(x),n,2))
    x_new = np.zeros((n*len(X),t,t,1))
    for i in range(len(x)):
        for j in range(n):
            x_new[i+j*len(X)] = x[i,indices[i,j,0]:indices[i,j,0]+t*s:s,indices[i,j,1]:indices[i,j,1]+t*s:s]
    return x_new

x_ = random_cut(x,3,48,1)

x_train, x_test = train_test_split(x_, test_size=0.2)



# Define attribute erasing operations
def raboter(X):
    X_ = X.copy()
    with np.nditer(X_, op_flags=['readwrite'], flags=['external_loop']) as it:
        for x in it:
            x[...] = np.minimum(x,0.8)
    return X_

def cut_paste(x):
    x_new = np.copy(x)
    s = np.random.randint(10,15,len(x))
    for i in range(len(x)):
        i_max = 48-s[i]
        indices_start = np.random.randint(0,i_max,2)
        indices_end = np.random.randint(0,i_max,2)
        for j in range(s[i]):
                for k in range(s[i]):
                    x_new[i,indices_end[0]+j,indices_end[1]+k] = x[i,indices_start[0]+j,indices_start[1]+k]
    return x_new

def saltNpepper(x,amount = 0.03):
      #row,col,ch = x.shape
      s_vs_p = 0.5
      out = np.copy(x)
      # Salt mode
      num_salt = np.ceil(amount * x.size * s_vs_p)
      coords = [np.random.randint(0, i, int(num_salt))
              for i in x.shape]
      out[tuple(coords)] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* x.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i, int(num_pepper))
              for i in x.shape]
      out[tuple(coords)] = 0
      return out

def boucher(X):
    X_ = X.copy()
    with np.nditer(X_, op_flags=['readwrite'], flags=['external_loop']) as it:
        for x in it:
            x[...] = np.maximum(x,0.2)
    return X_

def decalage(x):
    indices = np.random.randint(0,3,(len(x),48))
    signes = np.random.randint(0,2,(len(x),48))
    out = x.copy()
    for i in range(len(x)):
        for j in range(48):
            if indices[i,j] != 0:
                if signes[i,j] == 0:
                    out[i,j,indices[i,j]::] = x[i,j,:-indices[i,j]:]
                    out[i,j,:indices[i,j]:] = x[i,j,-indices[i,j]::]
                else:
                    out[i,j,-indices[i,j]::] = x[i,j,:indices[i,j]:]
                    out[i,j,:-indices[i,j]:] = x[i,j,indices[i,j]::]
    return out

def decalage_y(x):
    out = np.rot90(x.copy(),1,(1,2))
    return np.rot90(decalage(out),-1,(1,2))

def perte(x, n = 7):
    x_new = np.copy(x)
    s = np.random.randint(4,9,(len(x),n))
    for i in range(len(x)):
        i_max = 48-s[i]
        indices = np.random.randint(0,i_max,(2,n))
        for j in range(n):
            x_new[i,indices[0,j]:indices[0,j]+s[i,j],indices[1,j]:indices[1,j]+s[i,j]] = 0.5
    return x_new

fonctions = [raboter, cut_paste, saltNpepper, boucher,  decalage, decalage_y, perte]

def destruction(x):
    return np.vstack((func(x) for func in fonctions))

def stacker(x, func = fonctions):
    arrays = [x for i in range(len(func))]
    return np.concatenate(arrays)

# Build attribute erasing model
from autoencoder import AutoEncoder
unet_filters = [8, 16, 32, 64]
patch_size = [48, 48]

es = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience=10, verbose =1)
reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", patience=5, verbose=1,factor=0.1)


x_train2=stacker(x_train)

x_train2_del = destruction(x_train)

model = AutoEncoder(1, 1, unet_filters, "sigmoid", image_size = patch_size, latent_dim=128).autoencoder
model.summary()

history = model.fit(x_train2_del, x_train2, epochs=50, batch_size=32, validation_split=0.2, shuffle=True, callbacks=[es, reducelr])

model.save("AEM_weights")

loss_value = model.evaluate(destruction(x_test), stacker(x_test), verbose=1)

"""
model = tf.keras.models.load_model('AEM_weights')
"""


""""
# Show training loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss by epoch - Attribute erasing')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='right')
plt.show()
"""


# Build model with memory module


temp_model = AutoEncoder(1, 1, unet_filters, "sigmoid", image_size = patch_size, latent_dim=128)

history = temp_model.fit(x_train, x_train, epochs=50, batch_size=32, validation_split=0.2, shuffle=True, callbacks=[es, reducelr])

model_mem, encoder_mem, decoder_mem = temp_model.autoencoder, temp_model.encoder, temp_model.decoder
model_mem.save('mem_weights')
encoder_mem.save('mem_weights_enc')
decoder_mem.save('mem_weights_dec')

loss_value = model.evaluate(destruction(x_test), stacker(x_test), verbose=1)

"""
model_mem = tf.keras.models.load_model('mem_weights')
encoder_mem = tf.keras.models.load_model('mem_weights_enc')
decoder_mem = tf.keras.models.load_model('mem_weights_dec')
"""


"""
# Show training loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss by epoch - Attribute erasing')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='right')
plt.show()
"""

def exp_cos_sim(z,m):
    return np.exp(np.dot(z,m)/(np.linalg.norm(z,2)*np.linalg.norm(m,2)))

def HSSA(w,lamb):
    return np.where(w > lamb, w, np.zeros_like(w))

def predict_mem(x_tr,x_te,enc,dec,n=200):
    rng = np.random.default_rng()
    numbers = rng.choice(len(x_tr), size=n, replace=False)
    m = x_tr[numbers]
    latent_m = enc.predict(m)
    latent_te = enc.predict(x_te)
    latent_mem = np.zeros_like(latent_te)
    uu = np.zeros((len(x_te),n))
    for i in range(len(x_te)):
        for j in range(n):
            uu[i,j] = exp_cos_sim(latent_te[i],latent_m[j])
        uu[i] = HSSA(uu[i]/np.sum(uu[i]),1/n)
        uu[i] = uu[i]/np.linalg.norm(uu[i],1)
        latent_mem[i] = np.dot(uu[i],latent_m)
    pred = dec.predict(latent_mem)
    return pred



"""
preds = predict_mem(x_train,x_test,encoder_mem,decoder_mem, len(x_train)//10)

f, ax = plt.subplots(2,2)
f.set_size_inches(20, 20)
for i in range(2):
    ax[0,i].imshow(x_test[i].reshape(48, 48),cmap='gray')
    ax[1,i].imshow(preds[i].reshape(48, 48),cmap='gray')
plt.show()
"""


# Anomaly score with attribute erasing module

# Mean anomaly on training set
preds = model.predict(destruction(x_train))
x_start = x_train.reshape((-1,2304))
x_end = preds.reshape((-1,2304))

mean_outlier_score = np.zeros(len(fonctions))
for i in range(len(fonctions)):
    mean_outlier_score[i] = np.mean(np.linalg.norm(x_start-x_end[i*len(x_train):(i+1)*len(x_train)], ord=1, axis=1))
mean_outlier_score

# Anomaly score taking the mean anomaly on training set into account
def anomaly_score(x):
    x_test = destruction(x)
    preds = model.predict(x_test).reshape((-1,2304))
    score = np.zeros(len(x))
    for i in range(len(fonctions)):
        score += np.linalg.norm(x.reshape(-1,2304)-preds[i*len(x):(i+1)*len(x)], ord=1, axis=1)/mean_outlier_score[i]
    return score/len(fonctions)



# Anomaly score with memory module
mean_score_mem = np.mean(np.linalg.norm(x_train.reshape((-1,2304))-predict_mem(x_train,x_train,encoder_mem,decoder_mem,n=100).reshape((-1,2304)), ord=1, axis=1))

def anomaly_score_mem(x):
    pred = predict_mem(x_train,x,encoder_mem,decoder_mem,n=100)
    sc = np.linalg.norm(x.reshape((-1,2304))-pred.reshape((-1,2304)), ord=1, axis=1)/mean_score_mem
    return sc



# Display anomaly scores on test samples
def show_test_score(X,title = ''):
    plt_1 = plt.figure(figsize=(10, 5))
    plt.title('Anomaly score on the test sample'+title, size=15)
    plt.plot(np.arange(len(X)),X,label = "anomaly score")
    plt.plot(np.arange(len(X)),np.zeros_like(X)+X.mean(),label="global mean anomaly score")
    plt.xlabel("Samples",size=15)
    plt.ylabel("Anomaly score",size=15)
    plt.legend()
    plt.show()

show_test_score(anomaly_score(x_test),' - attribute erasing')
show_test_score(anomaly_score_mem(x_test),' - memory module')



# Display anomaly scores on error samples
def faille(X):
    i_rot = np.random.randint(0,4)
    x = 0
    y = np.random.randint(0,len(X)//2)
    dy = 0
    out = np.rot90(X.copy().reshape(len(X),len(X)),i_rot,(0,1))
    while x<48:
        if dy==0:
            dy = np.random.randint(-3,4)
        elif dy<0:
            y = max(0,y-1)
            dy+=1
            out[x,:y]=0
            x+=1
        else:
            y = min(47,y+1)
            dy+=-1
            out[x,:y]=0
            x+=1
    return np.rot90(out,-i_rot,(0,1)).reshape(len(X),len(X),1)

def griffure(X):
    i_rot = np.random.randint(0,4)
    x = 0
    y = np.random.randint(2*len(X)//5,3*len(X)//5+1)
    y2 = 0
    dy = 0
    dy2 = np.random.randint(2,4)
    out = np.rot90(X.copy().reshape(len(X),len(X)),i_rot,(0,1))
    while x < 47:
        if dy==0:
            dy = np.random.randint(-2,3)
        elif dy<0:
            y = max(0,y-1)
            y2 = max(0,y+dy2)
            dy+=1
            out[x:x+2,y:y2]=0
            x+=2
        else:
            y = min(47,y+1)
            y2 = max(0,y+dy2)
            dy+=-1
            out[x:x+2,y:y2]=0
            x+=2
    return np.rot90(out,-i_rot,(0,1)).reshape(len(X),len(X),1)

def incrustation(X):
    x_new = np.copy(X)
    s = np.random.randint(10,20)
    i_max = 48-s
    indices = np.random.randint(0,i_max,2)
    x_new[indices[0]:indices[0]+s,indices[1]:indices[1]+s] = 0.5
    return x_new

def randomisation(X):
    return decalage(decalage_y(X.reshape(1,48,48,1))).reshape(48,48,1)

default = [faille,griffure,incrustation,randomisation]

def anomaly_ord(X):
    i_def = np.random.randint(0,len(default),len(X))
    out = np.zeros((len(default)*len(X),48,48,1))
    for i in range(len(default)):
        for j in range(len(X)):
            out[i*len(X)+j]=default[i](X[j])
    return out.reshape(-1,48,48,1)

x_anomaly = anomaly_ord(x_test.copy())

def local_mean_score(X,sep=1):
    n = len(X)//sep
    out = np.ones_like(X)
    for i in range(sep):
        out[i*n:(i+1)*n] = X[i*n:(i+1)*n].mean()
    return out

def show_anomaly_score(X,title=''):
    plt_1 = plt.figure(figsize=(10, 5))
    plt.title('Anomaly score on the error sample'+title,size=15)
    plt.plot(np.arange(len(X)),X,label = "anomaly score")
    plt.plot(np.arange(len(X)),np.zeros_like(X)+X.mean(),label="global mean anomaly score")
    plt.plot(np.arange(len(X)),local_mean_score(X,len(default)),label="local mean anomaly score",color="black")
    plt.xlabel("Samples",size=15)
    plt.ylabel("Anomaly score",size=15)
    plt.legend()
    plt.show()

show_anomaly_score(anomaly_score(x_anomaly),' - attribute erasing')
show_anomaly_score(anomaly_score_mem(x_anomaly),' - memory module')



# Display anomaly scores on real samples
fnames = glob.glob('real_data/*')
arrays = np.array([np.genfromtxt(f, skip_header=0,delimiter=',') for f in fnames])

def cut_random(X,s=1,t=48,n=1):
    if s == 1:
        x = X.copy()
    else:
        x = gaussian_blur(X.astype('float32'))
        x = X.copy()
    i_max = min(min(x[i].shape[0],x[i].shape[1]) for i in range(len(x)))-t*s
    indices = np.random.randint(0,i_max,(len(x),n,2))
    x_new = np.zeros((n*len(x),t,t))
    for i in range(len(x)):
        for j in range(n):
            x_new[n*i+j] = x[i][indices[i,j,0]:indices[i,j,0]+t*s:s,indices[i,j,1]:indices[i,j,1]+t*s:s]
    return x_new

def show_real_score(X,title = ''):
    plt_1 = plt.figure(figsize=(10, 5))
    plt.title('Anomaly score on the real sample'+title, size=15)
    plt.plot(np.arange(len(X)),X,label = "anomaly score")
    plt.plot(np.arange(len(X)),np.zeros_like(X)+X.mean(),label="global mean anomaly score")
    plt.plot(np.arange(len(X)),local_mean_score(X,10),label="local mean anomaly score",color="black")
    plt.xlabel("Samples",size=15)
    plt.ylabel("Anomaly score",size=15)
    plt.legend()
    plt.show()

x_real = normalize(cut_random(arrays,n=10,s=1))

show_real_score(anomaly_score(x_real),' - attribute erasing')
show_real_score(anomaly_score_mem(x_real),' - memory module')

x_reals = np.zeros((len(arrays)-1,901,901))
for i in range(1,len(arrays)):
    x_reals[i-1]=arrays[i][100:1001,100:1001]
x_reals = x_reals.reshape((len(arrays)-1,901,901,1))

def random_cut_complet(X,s_min=1, s_max=2, t=48,n=1):
    ns = s_max-s_min+1
    x_new = np.zeros((ns*n*len(X),t,t,1))
    for i in range(ns):
        x_new[i*n*len(X):(i+1)*n*len(X)] = random_cut(X,s_min+i,t,n)
    return x_new

def anomaly_score_complet(X,s_min=1,s_max=3,t=48,n=4):
    x = normalize(random_cut_complet(X,s_min,s_max,t,n))
    score = np.zeros(len(X))
    score_max = np.zeros(len(X))
    score_max_image = np.zeros((len(X),48,48,1))
    ns = s_max-s_min+1
    for i in range(ns):
        for j in range(n):
            score_local = anomaly_score(x[i*n*len(X)+j*len(X):i*n*len(X)+(j+1)*len(X)])
            score += np.nan_to_num(score_local,nan=1)
            for k in range(len(X)):
                if score_local[k]>score_max[k]:
                    score_max_image[k] = x[i*n*len(X)+j*len(X)+k]
            score_max = np.fmax(score_max,score_local)
    return score/(n*ns), score_max, score_max_image

def show_real_score_complet(X,X_max,title = ''):
    plt_1 = plt.figure(figsize=(10, 5))
    plt.title('Anomaly score on the real sample'+title, size=15)
    plt.scatter(np.arange(len(X)),X,label = "mean anomaly score",s=100)
    plt.scatter(np.arange(len(X)),X_max,label = "maximum anomaly score",s=60)
    plt.xlabel("Samples",size=15)
    plt.ylabel("Anomaly score",size=15)
    plt.legend()
    plt.show()

score_real_complet, score_max_real, score_max_real_image = anomaly_score_complet(x_reals,1,4,48,1000)

show_real_score_complet(score_real_complet,score_max_real,' - multiscale approach')



# Display anomaly scores on different samples
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

im_path = glob.glob('other/*.png')
error_arrays = np.array([rgb2gray(imageio.imread(f)) for f in im_path])

x_error = cut_random(error_arrays, n=50)

def show_different_score(X,title = ''):
    plt_1 = plt.figure(figsize=(10, 5))
    plt.title('Anomaly score on the different sample'+title, size=15)
    plt.plot(np.arange(len(X)),X,label = "anomaly score")
    plt.plot(np.arange(len(X)),np.zeros_like(X)+X.mean(),label="global mean anomaly score")
    plt.plot(np.arange(len(X)),local_mean_score(X,len(error_arrays)),label="local mean anomaly score",color="black")
    plt.xlabel("Samples",size=15)
    plt.ylabel("Anomaly score",size=15)
    plt.legend()
    plt.show()

show_different_score(anomaly_score(x_error),' - attribute erasing')
show_different_score(anomaly_score_mem(x_error),' - memory module')



def show_different_score_complet(X,X_max,title = ''):
    plt_1 = plt.figure(figsize=(10, 5))
    plt.title('Anomaly score on the different sample'+title, size=15)
    plt.scatter(np.arange(len(X)),X,label = "mean anomaly score",s=100)
    plt.scatter(np.arange(len(X)),X_max,label = "maximum anomaly score",s=60)
    plt.xlabel("Samples",size=15)
    plt.ylabel("Anomaly score",size=15)
    plt.legend()
    plt.show()

error = np.zeros((len(error_arrays),250,165))
for i in range(0,len(error_arrays)):
    error[i]=error_arrays[i][0:250,0:165]
error = error.reshape((len(error_arrays),250,165,1))

score_real_complet2, score_max_real2, score_max_real_image2 = anomaly_score_complet(error,1,3,48,300)

show_different_score_complet(score_real_complet2,score_max_real2,' - multiscale approach')
