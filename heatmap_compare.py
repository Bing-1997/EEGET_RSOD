import random
import shutil

import cv2
import os
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from skimage.metrics import structural_similarity
from skimage.transform import resize


def heatmap(data,outpath,Pixels_r=100):
        # Calculate the point density
    data_t = np.array(data)
    data_x = data_t[:, 0]
    data_y = data_t[:, 1]
    #data_z = data_t[:, 2]
    xy = np.vstack([data_x, data_y])
    z = gaussian_kde(xy)(xy)
    #print(z)
        # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    #data_x, data_y, z = data_x[idx], data_y[idx], z[idx]



    fig = plt.figure(figsize=(9.84,9.84))

    ax = fig.add_subplot()
    ax.patch.set_facecolor("w")
    plt.xticks([]), plt.yticks([])
    cmap_gray_r = cm.get_cmap('viridis').reversed()

    plt.scatter(data_x, data_y,c=z, s=Pixels_r,cmap=cmap_gray_r)
    #plt.show()
    plt.savefig(outpath,dpi=82, bbox_inches='tight',pad_inches=-0.1)#


def remove_nth_element(lst, indices_to_remove):
    # 检查索引n是否在列表的范围内
    # if 0 <= n < len(lst):
    #     # 使用列表切片创建一个新列表，不包含第n个元素
    #     return lst[:n] + lst[n+1:]
    # else:
    #     # 如果索引n超出范围，返回原始列表
    #     return lst
    new_list = [element for index, element in enumerate(lst) if index not in indices_to_remove]
    return new_list


def grey_scale(image):
    img_gray = image  #cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rows, cols = img_gray.shape
    flat_gray = img_gray.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    output = np.uint8(255 / (B - A) * (img_gray - A+0.1))
    return output

def create_dataset(in_path,files_one):
    dataset=[]
    for data_name in files_one:
        f1 = open(in_path + data_name, 'r')  # 打开原始数据
        data = [line.rstrip('\n').split(',') for line in f1]

        for i in range(0, len(data)):
            if (data[i][-2]=='Fixation')and(data[i][-3]=='Fixation'):
                X = (float(data[i][21]) + float(data[i][23])) / 2 - 420  # 3\5
                Y = (float(data[i][22]) + float(data[i][24])) / 2  # 4\6
                X = (X / 1080) * 800
                Y = (Y / 1080) * 800

                temp = [X,Y]
                dataset.append(temp)

    return dataset

def cctest(y, yhat):
    x_cc = y.flatten()
    y_cc = yhat.flatten()
    acc = np.corrcoef(x_cc, y_cc)[0, 1]
    return acc
def kltest(y, yhat):
    y_kl = y / ((np.sum(y)) * 1.0)
    yhat_kl = yhat / ((np.sum(yhat)) * 1.0)
    eps = 2.2204e-16
    acc = np.sum(y_kl * np.log(eps + y_kl / (yhat_kl + eps)))
    return acc
def nsstest(y, yhat):
    yhat_norm = (yhat - np.mean(yhat)) / np.std(yhat)
    coordlist = np.argwhere(y >= 128)   #阈值
    temp = []
    for coord in coordlist:
        temp.append(yhat_norm[coord[0], coord[1]])
    acc = np.mean(temp)
    return acc

def SSIMtest(y,yhat):
    y = resize(y, yhat.shape)
    ssim = structural_similarity(y,yhat)
    return ssim



def loss(img_result,groundtruth):
    cc = cctest(img_result,groundtruth)
    nss = nsstest(img_result,groundtruth)
    kl = kltest(img_result,groundtruth)
    ssim = SSIMtest(img_result,groundtruth)
    return cc,nss,kl,ssim


def compare(name):
    groundtruth = cv2.imread('./attention_map/' + name + '.png', cv2.IMREAD_GRAYSCALE)
    cc_temp = []
    kl_temp = []
    nss_temp = []
    ssim_temp = []
    files = os.listdir('./temp/')
    sub_num = len(files)
    for i in range(0, sub_num):
        img_result = cv2.imread('./temp/' + name[0:5] + '_'+str(i)+'.png', cv2.IMREAD_GRAYSCALE)
        cc, nss, kl, ssim = loss(img_result, groundtruth)
        cc_temp.append(cc)
        kl_temp.append(kl)
        nss_temp.append(nss)
        ssim_temp.append(ssim)

    cc_mean = np.mean(cc_temp)
    kl_mean = np.mean(kl_temp)
    nss_mean = np.mean(nss_temp)
    ssim_mean = np.mean(ssim_temp)


    return cc_mean,kl_mean,nss_mean,ssim_mean





if __name__ == "__main__":
    #name = '00011.jpg.csv'
    files = os.listdir('K:/object detection/RS-EEG_dataset/gaze_point')
    file_cc = open('./cc.txt', 'a')
    file_kl = open('./kl.txt', 'a')
    file_nss = open('./nss.txt', 'a')
    file_ssim = open('./ssim.txt', 'a')


    for i in range(0,1):   #刺激材料循环

        os.mkdir('K:/object detection/RS-EEG_dataset/temp/')
        name = files[i]
        #name='11830'
        print(name)
        in_path = 'K:/object detection/RS-EEG_dataset/gaze_point/'+name+'/'
        files1 = os.listdir(in_path)
        sub_num = len(files1)
        cc = [name]
        kl = [name]
        nss = [name]
        ssim = [name]
        data = create_dataset(in_path, files1)

        out_path = 'I:/object detection/RS-EEG_dataset/temp/' + name[0:5] + '.png'  # 输出路径
        heatmap(data, out_path, 50)  # 通过计算gaze point密度生成显著性图
        img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        img = 255 - img
        img = cv2.GaussianBlur(img, (151, 151), 0)
        output = grey_scale(img)
        cv2.imwrite(out_path, img)
        for list_length in range(0,18):   #剔除数量
            os.mkdir('K:/object detection/RS-EEG_dataset/temp/')
            try:
                for i in range(0, 1):            #循环20次
                    #随机剔除被试
                    random_list = random.sample(range(0, 20), list_length)
                    random_list.sort()

                    files_one = remove_nth_element(files1, random_list)
                    data = create_dataset(in_path,files_one)

                    out_path = 'K:/object detection/RS-EEG_dataset/temp/' + name[0:5] + '_'+str(list_length)+'.png'  # 输出路径
                    heatmap(data,out_path,50)           #通过计算gaze point密度生成显著性图
                    img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)

                    img = 255-img
                    img = cv2.GaussianBlur(img, (201, 201), 0)
                    output = grey_scale(img)
                    cv2.imwrite(out_path, img)
                cc_temp, kl_temp, nss_temp, \
                ssim_temp= compare(name)
                cc.append(cc_temp)
                kl.append(kl_temp)
                nss.append(nss_temp)
                ssim.append(ssim_temp)


                shutil.rmtree('K:/object detection/RS-EEG_dataset/temp/')

            except:
                shutil.rmtree('K:/object detection/RS-EEG_dataset/temp/')
                continue



        file_cc.write(','.join('%s' % a for a in cc))
        file_cc.write('\n')


        file_kl.write(','.join('%s' % a for a in kl))
        file_kl.write('\n')


        file_nss.write(','.join('%s' % a for a in nss))
        file_nss.write('\n')

        file_ssim.write(','.join('%s' % a for a in ssim))
        file_ssim.write('\n')









