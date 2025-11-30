import os
import sys
import argparse
# 补充缺失的导入（原代码使用了torch相关模块但未导入）
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import IMPClusterer

import faiss
from data import *
from utilities import *
from networks import *
import matplotlib.pyplot as plt
import numpy as np
from domain_bus import DomainBus
from tqdm import tqdm
from centroid import *
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.optimize import linear_sum_assignment

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch training",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # domains：修正路径为Windows格式，确保文件可找到
    parser.add_argument("--source", help="Source" ,default='C:/Users/46025/Desktop/复现论文/osda2019/Reserve_to_Adapt/data/amazon_0-9_train_all.txt')
    parser.add_argument("--target", help="Target", default='C:/Users/46025/Desktop/复现论文/osda2019/Reserve_to_Adapt/data/webcam_0-9_20-30_test.txt')
    
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")   
   
    # number of classes: known, unknown and the classes of self-sup task
    parser.add_argument("--shared_classes", type=int, default=10, help="Number of classes of source domain -- known classes")
    parser.add_argument("--all_classes", type=int, default=12,help=" Known+unknown classes")
   
    # path of the folders used：修正数据根目录为本地路径
    parser.add_argument("--log_dir", default="C:/Users/46025/Desktop/复现论文/osda2019/Reserve_to_Adapt/log/of31/", help="Path of the log folder")
    parser.add_argument("--data_dir", default="C:/Users/46025/Desktop/复现论文/osda2019/Reserve_to_Adapt/data/", help="Path of the dataset")

    # to select gpu/num of workers
    parser.add_argument("--gpu", type=int, default=0, help="gpu chosen for the training")
  
    parser.add_argument("--use_VGG", action='store_true', default=False, help="If use VGG")
    parser.add_argument("--name", type=str, default='1')

    return parser.parse_args()

args = get_args()

orig_stdout = sys.stdout
max_iter = 10000
warmiter = 3 # a2d:2 for best result

args.log_dir = args.log_dir + args.source.split('/')[-1][0]+'2'+args.target.split('/')[-1][0]+'_'+args.name

# 修正日志目录创建：支持多级目录创建，避免权限问题
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir, exist_ok=True)

print('\n')    
print('TRAIN START!')
print('\n')
print('THE OUTPUT IS SAVED IN A TXT FILE HERE -------------------------------------------> ', args.log_dir)
print('\n')

f = open(args.log_dir + '/out.txt', 'w')
sys.stdout = f

# 源域数据转换：保持原逻辑，仅修正函数名避免重复定义
#标签one-hot编码
#特征从224x224x3转换为3x224x224张量
def transform_source(data, label, is_train):
    label = one_hot(args.all_classes, label)
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),#强制缩放到256x256
        transforms.RandomCrop(224),#随机裁取224x224
        transforms.RandomHorizontalFlip(),#随机水平翻转
        transforms.ToTensor(),#转换为张量 形状为[3,224,224],一张图片某个像素的 RGB 值是 (255, 127, 0)，经过 ToTensor() 后会变成 (1.0, 0.5, 0.0)
    ])
    data = transform_train(data)
    return data, label

images,labels = get_split_dataset_info(args.source, args.data_dir) #读取源域数据
ds = CustomDataset(images,labels,img_transformer=transform_source,is_train=True) #ds是一个Dataset对象
# 修正num_workers=0，避免Windows多线程错误
source_train = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)#droplast保证每次取一批数据，可能要BN?


# 目标域训练集转换：修正函数名避免重复定义 
# 给未知类留标签，0-9为已知类，10为未知类
def transform_target_train(data, label, is_train):
    if label in range(10):
        label = one_hot(11, label)
    else:
        label = one_hot(11,10)
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    data = transform_train(data)
    return data, label

images,labels = get_split_dataset_info(args.target, args.data_dir)
ds1 = CustomDataset(images,labels,img_transformer=transform_target_train,is_train=True)
# 修正num_workers=0，避免Windows多线程错误
target_train = torch.utils.data.DataLoader(ds1, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

# 目标域测试集转换：修正函数名避免重复定义
def transform_target_test(data, label, is_train):
    label = one_hot(31,label)
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    data = transform_test(data)
    return data, label

ds2 = CustomDataset(images,labels,img_transformer=transform_target_test,is_train=True)
# 修正num_workers=0，避免Windows多线程错误
target_test = torch.utils.data.DataLoader(ds2, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)

# ----------------------------load the class centroids bank
all_centroids = Centroids(class_num=args.shared_classes, dim=args.shared_classes, use_cuda=torch.cuda.is_available())#centroids对象，存储源域目标域共享类的类中心
discriminator = LargeAdversarialNetwork(256).cuda() if torch.cuda.is_available() else LargeAdversarialNetwork(256)
"""
对抗判别器LargeAdversarialNetwork，经特征提取、瓶颈层转为256维特征后输入对抗判别器
核心：梯度反转层GRL，在反向传播时将梯度符号取反，从而使特征提取器学习到对域不变的特征
同时优化特征提取器和对抗判别器，使得特征提取器生成的特征无法被对抗判别器区分源域和目标域
判别器更能分清楚源域和目标域特征
特征提取器学习到的特征对域不变，有助于分类器在目标域上表现更好
"""
# 修正ResNet模型路径为Windows本地缓存路径
feature_extractor = ResNetFc(model_name='resnet50',model_path='C:/Users/46025/Desktop/复现论文/osda2019/Reserve_to_Adapt/预训练model/resnet50-19c8e357.pth')
cls = CLS(feature_extractor.output_num(), args.all_classes, bottle_neck_dim=256)#分类器
net = nn.Sequential(feature_extractor, cls).cuda() if torch.cuda.is_available() else nn.Sequential(feature_extractor, cls)

# ----------------------------find virtual class
customgenearator = DomainBus([source_train, target_train])#迭代产生ource_train, target_train的数据对
with torch.no_grad():
    with Accumulator(['fs','ft','ls', 'lt']) as ProbRecorder:#feature_source, feature_target, label_source, label_target收集numpy数据的容器
        for i, ((im_source, label_source), (im_target, label_target)) in enumerate(customgenearator):#按批次读取数据对
            # 动态判断是否使用GPU
            if torch.cuda.is_available():
                im_source = im_source.cuda()
                label_source = label_source.cuda()
                im_target = im_target.cuda()
                label_target = label_target.cuda()
            
            _, feature_source, fc_source, predict_prob_source = net.forward(im_source)#源域数据前向传播得到：_:特征提取器输出的，feature_source:特征,维度是瓶颈层输出的256，fc_source:分类器输出(fc层)，predict_prob_source:预测概率(softmax)
            ft1, feature_target, fc_target, predict_prob_target = net.forward(im_target)#目标域数据前向传播,ft1：特征提取器输出的，feature_target:特征,维度是瓶颈层输出的256，fc_target:分类器输出(fc层)，predict_prob_target:预测概率(softmax)
            fs,ft,ls,lt = [variable_to_numpy(x) for x in (feature_source, feature_target, torch.nonzero(label_source,as_tuple=True)[1], torch.nonzero(label_target,as_tuple=True)[1]) ]#把收集到的数据转成numpy格式，以使用accumulator存储
            ProbRecorder.updateData(globals())#fs: np.ndarray [B,256]ft: np.ndarray [B,256]ls: np.ndarray [B]（源标签索引）lt: np.ndarray [B]（目标标签索引，训练时目标标签被映射/合并为11类等）讲这些追加到容器中
    #所有批次数据的结果已存储在ProbRecorder中
    s_centroids = [] # calculate source class centroids计算源域类中心
    for i in range(args.shared_classes):
        s_centroids.append(ProbRecorder['fs'][ProbRecorder['ls']==i].mean(axis=0))
    s_centroids = np.stack(s_centroids,axis=0)#根据源域标签索引ls，把对应的特征fs取出来，计算每个类的均值作为类中心，最终得到形状为[shared_classes,256]的s_centroids数组

    K_cluster =20# cluster target class centroids 对目标域类中心进行聚类
    faiss_kmeans = faiss.Kmeans(256, int(K_cluster), niter=800, verbose=False, min_points_per_centroid=1, gpu=False)#kmeans聚类
    faiss_kmeans.train(ProbRecorder['ft'])#对目标域特征ft进行聚类，得到K_cluster个类中心
    t_centroids = faiss_kmeans.centroids#记录目标域类中心，形状为[K_cluster,256]
    
    # find nomatched target cluster
    cost = np.linalg.norm(s_centroids[:,None,:] -  t_centroids[None,:,:],axis=-1)#源域类中心和目标域类中心之间的欧氏距离矩阵，形状为[shared_classes,K_cluster]
    _,t_match = linear_sum_assignment(cost)#匈牙利算法，找到源域类中心和目标域类中心之间的最佳匹配，t_match是目标域类中心的索引数组，表示每个源域类中心对应的目标域类中心索引
    nomatch = []#记录未匹配的目标域类中心
    for i in range(K_cluster):
        if i not in t_match:
            nomatch.append(t_centroids[i])#把未匹配的目标域类中心添加到nomatch列表中
    nomatch = np.stack(nomatch,axis=0)#把未匹配的类中心堆叠成数组，形状为[num_nomatch,256],定义为未知类
    
    fcweight = np.concatenate([s_centroids,nomatch],axis=0)#把源域类中心和未匹配的目标域类中心连接起来，作为分类器的初始权重，形状为[20,256]
    for key, v in net.state_dict().items():   # fast initial classifier weight  
        if key=='1.main.1.2.weight':
            cost = np.linalg.norm(fcweight[:,None,:] -  v.cpu().numpy()[None,:,:],axis=-1)
            _,t_match = linear_sum_assignment(cost)
            param = torch.from_numpy(v.cpu().numpy()[t_match]).cuda().detach().clone() if torch.cuda.is_available() else torch.from_numpy(v.cpu().numpy()[t_match]).detach().clone()
            net.state_dict()['1.fc.weight'].copy_(param)  
    """
    已经得到了聚类中心，fcweight = np.concatenate([s_centroids,nomatch],axis=0)拼成了分类器的初始权重
    这里和分类器的权重v进行了一次最佳匹配，然后把分类器的权重按照这个匹配重新排序，赋值给分类器的权重
    使得fc层的第i个输出对应于第i个类中心
    """
    nomatch = torch.from_numpy(nomatch).cuda().detach().clone() if torch.cuda.is_available() else torch.from_numpy(nomatch).detach().clone()
    #未匹配的目标域类中心，作为未知类的表示
del(ProbRecorder)

# 优化器配置
scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=max_iter)#逆衰减，学习率随训练步数增加而减小
optimizer_discriminator = OptimWithSheduler(optim.SGD(discriminator.parameters(), lr=args.learning_rate*10, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)#对抗判别器的优化器，采用随机梯度下降，学习率是基础学习率的10倍，权重衰减（L2正则化）5e-4，动量0.9，使用Nesterov加速梯度
optimizer_feature_extractor = OptimWithSheduler(optim.SGD(feature_extractor.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)#特征提取器的优化器，学习率是基础学习率,其他参数同上
optimizer_cls = OptimWithSheduler(optim.SGD(cls.parameters(), lr=args.learning_rate*10, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)#分类器的优化器，学习率是基础学习率的10倍，其他参数同上

## ----------------------------train start
epoch = 0
k=0
best_os = 0
best_os_star = 0
best_unk = 0
best_hos = 0
best_epoch = 0
# c_weight = torch.ones(args.shared_classes)

while epoch <70:#训练70个epoch
    customgenearator = DomainBus([source_train, target_train])#迭代产生ource_train, target_train的数据对
    losscounter = LossCounter()#记录各类损失的容器
    with Accumulator(['pred_s','pred_t','label_s', 'kl','fss','ftt']) as ProbRecorder:#feature_source, feature_target, label_source, label_target收集numpy数据的容器
        for i, ((im_source, label_source), (im_target, label_target)) in enumerate(customgenearator):
            # 动态判断是否使用GPU
            if torch.cuda.is_available():
                im_source = im_source.cuda()
                label_source = label_source.cuda()
                im_target = im_target.cuda()
                label_target = label_target.cuda()
            
            _, feature_source, fc_source, predict_prob_source = net.forward(im_source)#net只包含特征提取器和分类器CLS
            ft1, feature_target, fc_target, predict_prob_target = net.forward(im_target)#目标域数据前向传播,ft1：特征提取器输出的，feature_target:特征,维度是瓶颈层输出的256，fc_target:分类器输出(fc层)，predict_prob_target:预测概率(softmax)
            
            domain_prob_discriminator_1_source = discriminator.forward(feature_source)#对抗判别器对源域特征进行前向传播，得到源域的域概率分布
            domain_prob_discriminator_1_target = discriminator.forward(feature_target)#对抗判别器对目标域特征进行前向传播，得到目标域的域概率分布
            
            s_ctds, t_ctds = all_centroids.get_centroids()#获取源域和目标域的类中心
            _, pseudo_t_label = predict_prob_target[:,:args.shared_classes].max(1)#预测目标域样本的伪标签，取预测概率最大的类作为伪标签
            
            kltarget = torch.nn.functional.kl_div( (nn.Softmax(-1)(fc_target[:,:args.shared_classes])).log(),   s_ctds[pseudo_t_label], reduction='none').sum(1).detach()#计算目标域样本的softmax概率分布与为标签对应的源域类中心之间的KL散度，作为目标域样本的可信度指标
            kltarget = torch.where(torch.isinf(kltarget), torch.full_like(kltarget, 10), kltarget)#把inf值替换为10

            if epoch<=1:
                gmm = GaussianMixture(n_components=3, covariance_type='full').fit(to_np(kltarget)[:,None])#前两轮 对kld高斯混合模型、3个成分、均值最小：已知类、均值最大：未知类
            
            known_cluster = np.argmin(gmm.means_)#已知类对应的高斯成分索引
            unknown_cluster = np.argmax(gmm.means_)#未知类对应的高斯成分索引
            gmm_index = gmm.predict(to_np(kltarget)[:,None])#预测每个目标域样本属于哪个高斯成分，输出三维概率
            
            pred_s, pred_t, label_s, kl, fss, ftt \
                = [variable_to_numpy(x)  for x in (nn.Softmax(-1)(fc_source[:,:args.shared_classes]), \
                      predict_prob_target, label_source, kltarget, feature_source, feature_target)]#pred_s为源域样本预测概率，pred_t为目标域样本预测概率，label_s为源域样本标签索引，kl为目标域样本的kld可信度指标，fss为源域样本特征，ftt为目标域样本特征
            ProbRecorder.updateData(globals())#把这些追加到容器中

            weight = gmm.predict_proba(to_np(kltarget)[:,None])[:,known_cluster]#目标域样本属于已知类的概率
            weight = torch.tensor(weight).cuda().detach() if torch.cuda.is_available() else torch.tensor(weight).detach()#转换为张量
            
            if epoch<=10:# 前11轮训练取高可信度样本（高度属于已知类）
                weight = torch.where(weight>0.8,torch.tensor([1]).float().cuda(),torch.tensor([0]).float().cuda()).detach() if torch.cuda.is_available() else torch.where(weight>0.8,torch.tensor([1]).float(),torch.tensor([0]).float()).detach()#只有80%以上概率的样本才被认为是已知类样本
                r = torch.nonzero(torch.tensor(gmm_index!=known_cluster).cuda()).unsqueeze(-1) if torch.cuda.is_available() else torch.nonzero(torch.tensor(gmm_index!=known_cluster)).unsqueeze(-1)#不属于已知类高斯成分的样本
                topk=16#预设16
                if r.size()[0]>topk:#当不属于已知类高斯成分的样本数大于topk时，选取kld最大的topk个样本作为未知类样本
                    r = torch.sort(kltarget.detach(),dim = 0)[1][-1*topk:]
            else: # 11轮直接按之前的高斯成分划分            
                weight = torch.where(torch.tensor(gmm_index==known_cluster).cuda(),torch.tensor([1]).float().cuda(),torch.tensor([0]).float().cuda()).detach() if torch.cuda.is_available() else torch.where(torch.tensor(gmm_index==known_cluster),torch.tensor([1]).float(),torch.tensor([0]).float()).detach()
                r = torch.nonzero(torch.tensor(gmm_index==unknown_cluster).cuda()).unsqueeze(-1) if torch.cuda.is_available() else torch.nonzero(torch.tensor(gmm_index==unknown_cluster)).unsqueeze(-1)
            
            feature_otherep = torch.index_select(ft1, 0, r.view(-1))#按照r从目标域特征ft1中选取未知类样本的特征
            if r.size()[0]>1:#当未知类样本数大于1时，进行伪标签生成和交叉熵损失计算
                _, feature_otherep, logits_otherep, predict_prob_otherep = cls.forward(feature_otherep)#根据特征计算分类器输出和预测概率
                _, pseudo_index = predict_prob_otherep[:,args.shared_classes:].max(1)#取预测概率最大的类作为伪标签索引
                pseudo_index=pseudo_index + args.shared_classes#更新为全局为标签，避免与已知类标签冲突
                if torch.cuda.is_available():
                    pseudo_label = torch.zeros(r.size()[0],args.all_classes).cuda().scatter_(1,pseudo_index.unsqueeze(1),torch.ones(r.size()[0],1).cuda())#生成one-hot伪标签
                else:
                    pseudo_label = torch.zeros(r.size()[0],args.all_classes).scatter_(1,pseudo_index.unsqueeze(1),torch.ones(r.size()[0],1))
                ce_ep = CrossEntropyLoss(pseudo_label[:,:],predict_prob_otherep[:,:])#计算未知类交叉熵损失            
            else:#未知类样本数不足1时，交叉熵损失设为0
                ce_ep=torch.tensor(0.0).cuda() if torch.cuda.is_available() else torch.tensor(0.0)
               
            ce = CrossEntropyLoss(label_source, nn.Softmax(-1)(fc_source))#计算源域样本的交叉熵损失

            virtual_predict_prob_source = cls.virt_forward( nomatch, feature_source, fc_source[:,:],torch.nonzero(label_source)[:,1],)#对源域样本进行虚拟类前向传播，nomatch为未知类的类中心，得到虚拟类的预测概率
            if torch.cuda.is_available():
                p = torch.zeros([label_source.shape[0],nomatch.size(0)]).cuda()#就是说0-9不管，之后的未知赋值为0因为是已知类
            else:
                p = torch.zeros([label_source.shape[0],nomatch.size(0)])
            v_label_source = torch.cat((label_source[:,:],p),1)#把源域样本的真实标签和虚拟类标签拼接起来，作为虚拟类的标签
            virtual_ce = CrossEntropyLoss(v_label_source, virtual_predict_prob_source)#虚拟类交叉熵损失
    
            entropy = EntropyLoss(predict_prob_target [:,:], instance_level_weight= weight.contiguous())#计算目标域样本的熵损失，使用权重weight调整样本贡献，也就是只关注高可信度样本

            if torch.cuda.is_available():#计算对抗损失
                adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source).cuda(), predict_prob=domain_prob_discriminator_1_source )
                adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target).cuda(), predict_prob=1 - domain_prob_discriminator_1_target, 
                                            instance_level_weight = weight.contiguous())#只关注目标域高可信度样本
            else:
                adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), predict_prob=domain_prob_discriminator_1_source )
                adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), predict_prob=1 - domain_prob_discriminator_1_target, 
                                            instance_level_weight = weight.contiguous())

            with OptimizerManager([optimizer_cls, optimizer_feature_extractor,optimizer_discriminator]):
                if epoch<=warmiter:#暖启阶段只用交叉熵和虚拟类损失
                    loss = 1 * ce + 1* virtual_ce + 0 * adv_loss + 0 * entropy + 0 * ce_ep         
                else:
                    loss = ce + 0.01 * virtual_ce + 0.3 * adv_loss + 1 * entropy + 1 * ce_ep 
                # if epoch<=warmiter: d2a
                #     loss = 1 * ce + 0* virtual_ce + 0 * adv_loss + 0 * entropy + 0 * ce_ep         
                # else:
                #     loss = ce + 0 * virtual_ce + 0.3 * adv_loss + 1 * entropy + 1 * ce_ep 
                loss.backward()#反向传播计算梯度
            losscounter.addOntBatch(ce, entropy, virtual_ce, ce_ep, adv_loss)#记录各类损失
            k += 1#迭代步数
            if torch.cuda.is_available():
                torch.cuda.empty_cache()#清理GPU缓存，释放显存

    all_centroids.update(ProbRecorder['pred_s'],ProbRecorder['pred_t'],ProbRecorder['label_s'])#用预测标签更新源域和目标域的类中心
    #一次训练结束
    # After train
    s_centroids = []
    for i in range(args.shared_classes):
        s_centroids.append(ProbRecorder['fss'][np.nonzero(ProbRecorder['label_s'])[1]==i].mean(axis=0))#用源域真实标签更新源域类中心
    s_centroids = np.stack(s_centroids,axis=0)

    faiss_kmeans = faiss.Kmeans(256, int(K_cluster), niter=800, verbose=False, min_points_per_centroid=1, gpu=False)#对目标域类中心进行聚类K_cluster = 20
    faiss_kmeans.train(ProbRecorder['ftt'])  #对目标域特征ft训练
    t_centroids = faiss_kmeans.centroids#记录目标域类中心，形状为[K_cluster,256]

    # find nomatched target cluster
    cost = np.linalg.norm(s_centroids[:,None,:] -  t_centroids[None,:,:],axis=-1)#源域类中心和目标域类中心之间的欧氏距离矩阵，形状为[shared_classes,K_cluster]
    _,t_match = linear_sum_assignment(cost)#匈牙利算法匹配到源域的已知类
    nomatch = []#记录未匹配的目标域类中心
    for i in range(K_cluster):
        if i not in t_match:#把未匹配的目标域类中心添加到nomatch列表中
            nomatch.append(t_centroids[i])
    nomatch = np.stack(nomatch,axis=0)
    nomatch = torch.from_numpy(nomatch).cuda().detach().clone() if torch.cuda.is_available() else torch.from_numpy(nomatch).detach().clone()

    if epoch ==warmiter:# warmiter结束时，重新初始化分类器权重
        # cluster shared class+K 
        faiss_kmeans = faiss.Kmeans(256, int(args.all_classes), niter=800, verbose=False, min_points_per_centroid=1, gpu=False)#对目标域类中心进行聚类K_cluster = all_classes
        faiss_kmeans.train(ProbRecorder['ftt'])#对目标域特征ft训练

        t_centroids = faiss_kmeans.centroids#记录目标域类中心，形状为[all_classes,256]
        cost = np.linalg.norm(s_centroids[:,None,:] -  t_centroids[None,:,:],axis=-1)#源域类中心和目标域类中心之间的欧氏距离矩阵，形状为[shared_classes,all_classes]
        _,t_match = linear_sum_assignment(cost)#匈牙利算法匹配到源域的已知类
        # no match as unk weight
        init_unk_weight = []#记录未匹配的目标域类中心
        for i in range(args.all_classes):
            if i not in t_match:
                init_unk_weight.append(t_centroids[i])
        init_unk_weight = np.stack(init_unk_weight,axis=0)
        
        for key, v in net.state_dict().items():   
            if key=='1.main.1.2.weight':
                v.requires_grad = False
                net.state_dict()['1.fc.weight'].requires_grad = False
                
                vvnorm = (torch.norm(v, dim = -1)).mean().cpu().numpy()#计算分类器权重的平均范数
                init_unk_weight = init_unk_weight/np.linalg.norm(init_unk_weight,axis=-1,keepdims=True)*vvnorm#归一化未匹配类中心的范数，使其与分类器权重的平均范数一致
                fcweight = np.concatenate([v[:args.shared_classes].clone().detach().cpu().numpy(), init_unk_weight,],axis=0)#拼接成新的分类器权重
                param = torch.from_numpy(fcweight).cuda().detach().clone() if torch.cuda.is_available() else torch.from_numpy(fcweight).detach().clone()#转换为张量
                net.state_dict()['1.fc.weight'].copy_(param)  
                
                v.requires_grad = True
                net.state_dict()['1.fc.weight'].requires_grad = True#恢复梯度计算
    
    if epoch<=30:#使用DPGMM对kld进行建模，后期只分两个簇
        gmm = BayesianGaussianMixture(n_components=4, max_iter=800).fit(ProbRecorder['kl'][:,None])
    else:
        gmm = BayesianGaussianMixture(n_components=2, max_iter=800).fit(ProbRecorder['kl'][:,None])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =================================evaluation
    with TrainingModeManager([feature_extractor, cls], train=False) as mgr, Accumulator(['predict_prob','predict_index', 'label']) as accumulator:
        for (i, (im, label)) in enumerate(target_test):
            if torch.cuda.is_available():
                im = im.cuda()
                label = label.cuda()
            ss, fs,_,  predict_prob = net.forward(im)
            predict_prob, label = [variable_to_numpy(x) for x in (predict_prob,label)]
            label = np.argmax(label, axis=-1).reshape(-1, 1)
            predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)
            accumulator.updateData(globals())
        
    for x in list(accumulator.keys()):
        globals()[x] = accumulator[x]

    y_true = label.flatten()
    y_pred = predict_index.flatten()
    m = extended_confusion_matrix(y_true, y_pred, true_labels=(list(range(args.shared_classes))+list(range(20,31))), pred_labels=list(range(args.all_classes)))

    cm = m
    cm = cm.astype(float) / np.sum(cm, axis=1, keepdims=True)
    acc_os_star = sum([cm[i][i] for i in range(args.shared_classes)]) / args.shared_classes
    unkn = sum(sum([cm[i][args.shared_classes:] for i in range(10, 21)])) / 11  
    acc_os = (acc_os_star * args.shared_classes + unkn) / 11
           
    hos = (2*acc_os_star*unkn)/(acc_os_star+unkn)                
    ce = losscounter.ce/losscounter.batch
    entropy = losscounter.entropy/losscounter.batch
    virtual = losscounter.virtual/losscounter.batch
    ce_ep = losscounter.ce_ep/losscounter.batch
    adv = losscounter.adv/losscounter.batch
    print ('Epoch:{}\tOS: {:.3f}\tOS*:{:.3f}\tUnk:{:.3f}\tHos:{:.3f}\tce: {:.3f}\tentropy:{:.3f}\tvirtual:{:.3f}\tce_ep:{:.3f}\tadv:{:.3f}'.format(epoch,acc_os,acc_os_star,unkn,hos, ce, entropy, virtual, ce_ep, adv))
    
    if hos>best_hos:
        best_os = acc_os
        best_os_star = acc_os_star
        best_unk = unkn
        best_hos = hos
        best_epoch = epoch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    epoch = epoch + 1

print ('Best: Epoch:{}\tOS: {:.3f}\tOS*:{:.3f}\tUnk:{:.3f}\tHos:{:.3f}'.format(best_epoch, best_os,best_os_star,best_unk,best_hos))
print('class_num'+ str(args.all_classes)   + str(args))
sys.stdout = orig_stdout
f.close()