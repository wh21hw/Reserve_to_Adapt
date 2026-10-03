import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
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
import shutil
from IMPClusterer import IMPClusterer
# 确保BASE_DIR正确获取（兼容Colab环境）
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()


def get_args():
    """伪代码环节：输入参数定义，新增lambda参数配置"""
    parser = argparse.ArgumentParser(
        description="开放集领域自适应训练脚本（支持lambda参数调整）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 路径配置
    parser.add_argument(
        "--source",
        help="源域数据路径",
        default=os.path.join(BASE_DIR, "data", "webcam_0-9_train_all.txt")
    )
    parser.add_argument(
        "--target",
        help="目标域数据路径",
        default=os.path.join(BASE_DIR, "data", "dslr_0-9_20-30_test.txt")
    )

    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")

    # 类别数量
    parser.add_argument("--shared_classes", type=int, default=10, help="源域已知类数量")
    parser.add_argument("--all_classes", type=int, default=12, help="总类别数（已知+未知）")

    # 超参数lambda（新增）
    parser.add_argument("--lambda1", type=float, default=0.01, help="虚拟损失L_vir的权重（λ₁）")
    parser.add_argument("--lambda2", type=float, default=1.0, help="未知类分离损失L_unk的权重（λ₂）")
    parser.add_argument("--lambda3", type=float, default=0.3, help="领域对抗损失L_adv的权重（λ₃）")

    # 文件夹路径
    parser.add_argument(
        "--log_dir",
        default=os.path.join(BASE_DIR, "log", "of31"),
        help="日志文件夹路径"
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join(BASE_DIR, "data") + "/",
        help="数据集根路径"
    )

    # GPU和模型选择
    parser.add_argument("--gpu", type=int, default=0, help="训练使用的GPU编号")
    parser.add_argument("--use_VGG", action='store_true', default=False, help="是否使用VGG作为特征提取器")
    parser.add_argument("--name", type=str, default='1', help="实验名称")

    return parser.parse_args()


args = get_args()

# 日志目录设置（包含lambda参数，方便区分实验）
args.log_dir = os.path.join(args.log_dir,
                          f"{args.source.split('/')[-1][0]}2{args.target.split('/')[-1][0]}_lambda1-{args.lambda1}_lambda2-{args.lambda2}_lambda3-{args.lambda3}_{args.name}")
os.makedirs(args.log_dir, exist_ok=True)

# 自定义日志类：同时输出到屏幕和文件
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 初始化日志
log_file = os.path.join(args.log_dir, "training_log.txt")
sys.stdout = Logger(log_file)

# 打印训练开始信息（包含lambda参数）
print('\nTRAIN START!')
print(f'超参数设置：lambda1={args.lambda1}, lambda2={args.lambda2}, lambda3={args.lambda3}')
print(f'THE OUTPUT IS SAVED IN: {args.log_dir}')
print(f'Log file: {log_file}\n')


# 数据转换函数
def transform_source(data, label, is_train):
    label = one_hot(args.all_classes, label)
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    data = transform_train(data)
    return data, label

def transform_target_train(data, label, is_train):
    if label in range(10):
        label = one_hot(11, label)
    else:
        label = one_hot(11, 10)
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    data = transform_train(data)
    return data, label

def transform_target_test(data, label, is_train):
    label = one_hot(31, label)
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    data = transform_test(data)
    return data, label


# 数据加载
# 源域数据
images, labels = get_split_dataset_info(args.source, args.data_dir)
images = [os.path.join(args.data_dir, im) for im in images]
ds = CustomDataset(images, labels, img_transformer=transform_source, is_train=True)
source_train = torch.utils.data.DataLoader(
    ds, batch_size=args.batch_size, shuffle=True,
    num_workers=4, pin_memory=True, drop_last=True
)

# 目标域训练集
images_t, labels_t = get_split_dataset_info(args.target, args.data_dir)
images_t = [os.path.join(args.data_dir, im) for im in images_t]
ds1 = CustomDataset(images_t, labels_t, img_transformer=transform_target_train, is_train=True)
target_train = torch.utils.data.DataLoader(
    ds1, batch_size=args.batch_size, shuffle=True,
    num_workers=4, pin_memory=True, drop_last=True
)

# 目标域测试集
ds2 = CustomDataset(images_t, labels_t, img_transformer=transform_target_test, is_train=True)
target_test = torch.utils.data.DataLoader(
    ds2, batch_size=args.batch_size, shuffle=True,
    num_workers=4, pin_memory=True, drop_last=False
)


# 模型初始化
use_cuda = torch.cuda.is_available()
all_centroids = Centroids(class_num=args.shared_classes, dim=args.shared_classes, use_cuda=use_cuda)

# 判别器
discriminator = LargeAdversarialNetwork(256)
if use_cuda:
    discriminator = discriminator.cuda()

# 特征提取器和分类器
feature_extractor = ResNetFc(
    model_name='resnet50',
    model_path=os.path.join(BASE_DIR, "预训练model", "resnet50-19c8e357.pth")
)
cls = CLS(feature_extractor.output_num(), args.all_classes, bottle_neck_dim=256)
net = nn.Sequential(feature_extractor, cls)
if use_cuda:
    net = net.cuda()

# 微调专用优化器（无调度器，固定学习率）
optimizer_feature_extractor = optim.SGD(
    feature_extractor.parameters(), 
    lr=args.learning_rate,
    weight_decay=5e-4, momentum=0.9, nesterov=True
)
optimizer_cls = optim.SGD(
    cls.parameters(), 
    lr=args.learning_rate * 10,
    weight_decay=5e-4, momentum=0.9, nesterov=True
)
# ====================== 【新增】源域微调代码（思路1核心） ======================
fine_tune_epochs = 5  # 可调整：3/5/7轮，先试5轮
net.train()  # 模型设为训练模式
print("===== 开始源域微调（优化第一阶段聚类特征） =====")
for ft_epoch in range(fine_tune_epochs):
    total_ft_loss = 0.0
    # 只遍历源域训练集，不处理靶域数据
    for im_source, label_source in tqdm(source_train, desc=f"源域微调Epoch {ft_epoch+1}/{fine_tune_epochs}"):
        if use_cuda:
            im_source = im_source.cuda()
            label_source = label_source.cuda()
        
        # 前向传播：仅计算源域分类输出
        _, feature_source, fc_source, predict_prob_source = net.forward(im_source)
        # 源域已知类分类损失（只取前10类，因为shared_classes=10）
        ce_loss = CrossEntropyLoss(label_source[:, :args.shared_classes], nn.Softmax(-1)(fc_source[:, :args.shared_classes]))
        
        # 反向传播：仅更新特征提取器和分类器，判别器不更新
        optimizer_feature_extractor.zero_grad()  # 特征提取器优化器清零
        optimizer_cls.zero_grad()                # 分类器优化器清零
        ce_loss.backward()                       # 损失回传
        optimizer_feature_extractor.step()       # 更新特征提取器参数
        optimizer_cls.step()                     # 更新分类器参数
        
        total_ft_loss += ce_loss.item()
    
    # 打印每轮微调损失（验证特征是否在优化）
    avg_ft_loss = total_ft_loss / len(source_train)
    print(f"源域微调Epoch {ft_epoch+1} | 平均已知类分类损失: {avg_ft_loss:.4f}")
print("===== 源域微调完成，开始虚拟类别发现与初始化 =====")
# ====================== 源域微调代码结束 ======================

# 虚拟类别发现与初始化
customgenearator = DomainBus([source_train, target_train])
with torch.no_grad():
    with Accumulator(['fs','ft','ls', 'lt']) as ProbRecorder:
        for i, ((im_source, label_source), (im_target, label_target)) in enumerate(customgenearator):
            if use_cuda:
                im_source = im_source.cuda()
                label_source = label_source.cuda()
                im_target = im_target.cuda()
                label_target = label_target.cuda()

            _, feature_source, fc_source, predict_prob_source = net.forward(im_source)
            ft1, feature_target, fc_target, predict_prob_target = net.forward(im_target)
            fs, ft, ls, lt = [variable_to_numpy(x) for x in (
                feature_source, feature_target,
                torch.nonzero(label_source, as_tuple=True)[1],
                torch.nonzero(label_target, as_tuple=True)[1]
            )]
            ProbRecorder.updateData(globals())

    # 计算源域类别中心
    s_centroids = []
    for i in range(args.shared_classes):
        s_centroids.append(ProbRecorder['fs'][ProbRecorder['ls'] == i].mean(axis=0))
    s_centroids = np.stack(s_centroids, axis=0)

    # 聚类目标域特征
    #K_cluster = 20
    
    #faiss_kmeans = faiss.Kmeans(256, int(K_cluster), niter=800, verbose=False,
    #                           min_points_per_centroid=1, gpu=False)
    #faiss_kmeans.train(ProbRecorder['ft'])
    #t_centroids = faiss_kmeans.centroids
    imp = IMPClusterer(alpha = 0.05)
    #获取目标域特征
    ft_tensor = torch.from_numpy(ProbRecorder['ft']).cuda()
    #进行imp聚类
    t_centroids_tensor = imp.fit(
      features=ft_tensor,
      source_centroids=torch.from_numpy(s_centroids).cuda(),
      use_source_centroids=True,
      fix_source_centroids=True
    )
    t_centroids = t_centroids_tensor.cpu().numpy()
    print(f"初始化聚类簇数为{len(t_centroids)}")
    K_cluster = len(t_centroids)
    # 找到不匹配的目标域聚类
    cost = np.linalg.norm(s_centroids[:, None, :] - t_centroids[None, :, :], axis=-1)
    _, t_match = linear_sum_assignment(cost)
    nomatch = []
    for i in range(K_cluster):
        if i not in t_match:
            nomatch.append(t_centroids[i])
    nomatch = np.stack(nomatch, axis=0)

    # 初始化分类器权重
    fcweight = np.concatenate([s_centroids, nomatch], axis=0)
    for key, v in net.state_dict().items():
        if key == '1.main.1.2.weight':
            cost = np.linalg.norm(fcweight[:, None, :] - v.cpu().numpy()[None, :, :], axis=-1)
            _, t_match = linear_sum_assignment(cost)
            param = torch.from_numpy(v.cpu().numpy()[t_match])
            if use_cuda:
                param = param.cuda()
            net.state_dict()['1.fc.weight'].copy_(param)

    nomatch = torch.from_numpy(nomatch)
    if use_cuda:
        nomatch = nomatch.cuda()
    nomatch = nomatch.detach().clone()

del ProbRecorder


# 优化器配置
max_iter = 10000
warmiter = 3

scheduler = lambda step, initial_lr: inverseDecaySheduler(
    step, initial_lr, gamma=10, power=0.75, max_iter=max_iter
)

# 只定义判别器优化器 + 复用微调的特征/分类器优化器（加调度器）
optimizer_discriminator = OptimWithSheduler(
    optim.SGD(discriminator.parameters(), lr=args.learning_rate*10,
             weight_decay=5e-4, momentum=0.9, nesterov=True),
    scheduler
)
optimizer_feature_extractor = OptimWithSheduler(optimizer_feature_extractor, scheduler)
optimizer_cls = OptimWithSheduler(optimizer_cls, scheduler)


## 训练主循环
epoch = 0
k = 0
best_os = 0
best_os_star = 0
best_unk = 0
best_hos = 0
best_epoch = 0

while epoch < 70:
    customgenearator = DomainBus([source_train, target_train])
    losscounter = LossCounter()
    with Accumulator(['pred_s','pred_t','label_s', 'kl','fss','ftt']) as ProbRecorder:
        for i, ((im_source, label_source), (im_target, label_target)) in enumerate(customgenearator):
            if use_cuda:
                im_source = im_source.cuda()
                label_source = label_source.cuda()
                im_target = im_target.cuda()
                label_target = label_target.cuda()

            _, feature_source, fc_source, predict_prob_source = net.forward(im_source)
            ft1, feature_target, fc_target, predict_prob_target = net.forward(im_target)

            domain_prob_discriminator_1_source = discriminator.forward(feature_source)
            domain_prob_discriminator_1_target = discriminator.forward(feature_target)

            s_ctds, t_ctds = all_centroids.get_centroids()
            _, pseudo_t_label = predict_prob_target[:, :args.shared_classes].max(1)

            # 计算KL散度
            kltarget = torch.nn.functional.kl_div(
                (nn.Softmax(-1)(fc_target[:, :args.shared_classes])).log(),
                s_ctds[pseudo_t_label], reduction='none'
            ).sum(1).detach()
            kltarget = torch.where(torch.isinf(kltarget), torch.full_like(kltarget, 10), kltarget)

            # GMM聚类初始化
            if epoch <= 1:
                gmm = GaussianMixture(n_components=3, covariance_type='full').fit(to_np(kltarget)[:, None])

            known_cluster = np.argmin(gmm.means_)
            unknown_cluster = np.argmax(gmm.means_)
            gmm_index = gmm.predict(to_np(kltarget)[:, None])

            # 记录概率和特征
            pred_s, pred_t, label_s, kl, fss, ftt = [variable_to_numpy(x) for x in (
                nn.Softmax(-1)(fc_source[:, :args.shared_classes]),
                predict_prob_target, label_source, kltarget, feature_source, feature_target
            )]
            ProbRecorder.updateData(globals())

            # 计算权重
            weight = gmm.predict_proba(to_np(kltarget)[:, None])[:, known_cluster]
            weight = torch.tensor(weight)
            if use_cuda:
                weight = weight.cuda()
            weight = weight.detach()

            # 选择样本
            if epoch <= 10:
                if use_cuda:
                    weight = torch.where(weight > 0.8, torch.tensor([1.0]).float().cuda(),
                                        torch.tensor([0.0]).float().cuda()).detach()
                else:
                    weight = torch.where(weight > 0.8, torch.tensor([1.0]).float(),
                                        torch.tensor([0.0]).float()).detach()
                r = torch.nonzero(torch.tensor(gmm_index != known_cluster))
                if use_cuda:
                    r = r.cuda()
                r = r.unsqueeze(-1)
                topk = 16
                if r.size()[0] > topk:
                    r = torch.sort(kltarget.detach(), dim=0)[1][-1 * topk:]
            else:
                if use_cuda:
                    weight = torch.where(torch.tensor(gmm_index == known_cluster).cuda(),
                                        torch.tensor([1.0]).float().cuda(),
                                        torch.tensor([0.0]).float().cuda()).detach()
                else:
                    weight = torch.where(torch.tensor(gmm_index == known_cluster),
                                        torch.tensor([1.0]).float(),
                                        torch.tensor([0.0]).float()).detach()
                r = torch.nonzero(torch.tensor(gmm_index == unknown_cluster))
                if use_cuda:
                    r = r.cuda()
                r = r.unsqueeze(-1)

            # 计算额外分类损失（未知类分离损失）
            feature_otherep = torch.index_select(ft1, 0, r.view(-1))
            if r.size()[0] > 1:
                _, feature_otherep, logits_otherep, predict_prob_otherep = cls.forward(feature_otherep)
                _, pseudo_index = predict_prob_otherep[:, args.shared_classes:].max(1)
                pseudo_index = pseudo_index + args.shared_classes
                if use_cuda:
                    pseudo_label = torch.zeros(r.size()[0], args.all_classes).cuda().scatter_(
                        1, pseudo_index.unsqueeze(1), torch.ones(r.size()[0], 1).cuda()
                    )
                else:
                    pseudo_label = torch.zeros(r.size()[0], args.all_classes).scatter_(
                        1, pseudo_index.unsqueeze(1), torch.ones(r.size()[0], 1)
                    )
                ce_ep = CrossEntropyLoss(pseudo_label[:, :], predict_prob_otherep[:, :])
            else:
                ce_ep = torch.tensor(0.0).cuda() if use_cuda else torch.tensor(0.0)

            # 计算各种损失
            ce = CrossEntropyLoss(label_source, nn.Softmax(-1)(fc_source))  # 源域已知类损失

            # 虚拟损失（L_vir）
            virtual_predict_prob_source = cls.virt_forward(
                nomatch, feature_source, fc_source[:, :],
                torch.nonzero(label_source)[:, 1],
            )
            if use_cuda:
                p = torch.zeros([label_source.shape[0], nomatch.size(0)]).cuda()
            else:
                p = torch.zeros([label_source.shape[0], nomatch.size(0)])
            v_label_source = torch.cat((label_source[:, :], p), 1)
            virtual_ce = CrossEntropyLoss(v_label_source, virtual_predict_prob_source)

            # 熵损失（L_ent）
            entropy = EntropyLoss(predict_prob_target[:, :], instance_level_weight=weight.contiguous())

            # 领域对抗损失（L_adv）
            if use_cuda:
                adv_loss = BCELossForMultiClassification(
                    label=torch.ones_like(domain_prob_discriminator_1_source).cuda(),
                    predict_prob=domain_prob_discriminator_1_source
                )
                adv_loss += BCELossForMultiClassification(
                    label=torch.ones_like(domain_prob_discriminator_1_target).cuda(),
                    predict_prob=1 - domain_prob_discriminator_1_target,
                    instance_level_weight=weight.contiguous()
                )
            else:
                adv_loss = BCELossForMultiClassification(
                    label=torch.ones_like(domain_prob_discriminator_1_source),
                    predict_prob=domain_prob_discriminator_1_source
                )
                adv_loss += BCELossForMultiClassification(
                    label=torch.ones_like(domain_prob_discriminator_1_target),
                    predict_prob=1 - domain_prob_discriminator_1_target,
                    instance_level_weight=weight.contiguous()
                )

            # 反向传播：使用命令行传入的lambda参数调整损失权重
            with OptimizerManager([optimizer_cls, optimizer_feature_extractor, optimizer_discriminator]):
                if epoch <= warmiter:
                    # 预热阶段：仅使用核心损失，权重固定
                    loss = 1 * ce + 1 * virtual_ce + 0 * adv_loss + 0 * entropy + 0 * ce_ep
                else:
                    loss = ce + \
                           args.lambda1 * virtual_ce + \
                           args.lambda3 * adv_loss + \
                           1 * entropy + \
                           args.lambda2 * ce_ep     
                loss.backward()

            losscounter.addOntBatch(ce, entropy, virtual_ce, ce_ep, adv_loss)
            k += 1
            if use_cuda:
                torch.cuda.empty_cache()

    # 更新中心
    all_centroids.update(ProbRecorder['pred_s'], ProbRecorder['pred_t'], ProbRecorder['label_s'])

    # 重新计算中心和聚类
    s_centroids = []
    for i in range(args.shared_classes):
        s_centroids.append(ProbRecorder['fss'][np.nonzero(ProbRecorder['label_s'])[1] == i].mean(axis=0))
    s_centroids = np.stack(s_centroids, axis=0)

    #faiss_kmeans = faiss.Kmeans(256, int(K_cluster), niter=800, verbose=False,
    #                           min_points_per_centroid=1, gpu=False)
    #faiss_kmeans.train(ProbRecorder['ftt'])
    #t_centroids = faiss_kmeans.centroids
    ftt_tensor = torch.from_numpy(ProbRecorder['ftt']).cuda()
    t_centroids_tensor = imp.fit(
      features=ftt_tensor,
      source_centroids=torch.from_numpy(s_centroids).cuda(),
      use_source_centroids=True,
      fix_source_centroids=True
    )
    t_centroids = t_centroids_tensor.cpu().numpy()
    K_cluster = len(t_centroids)
    print(f"第{epoch+1}轮训练后聚类簇数为{len(t_centroids)}")
    # 找到不匹配的目标域聚类
    cost = np.linalg.norm(s_centroids[:, None, :] - t_centroids[None, :, :], axis=-1)
    _, t_match = linear_sum_assignment(cost)
    nomatch = []
    for i in range(K_cluster):
        if i not in t_match:
            nomatch.append(t_centroids[i])
    nomatch = np.stack(nomatch, axis=0)
    nomatch = torch.from_numpy(nomatch)
    if use_cuda:
        nomatch = nomatch.cuda()
    nomatch = nomatch.detach().clone()

    # 初始化未知类权重
    if epoch == warmiter:
        faiss_kmeans = faiss.Kmeans(256, int(args.all_classes), niter=800, verbose=False,
                                   min_points_per_centroid=1, gpu=False)
        faiss_kmeans.train(ProbRecorder['ftt'])

        t_centroids = faiss_kmeans.centroids
        cost = np.linalg.norm(s_centroids[:, None, :] - t_centroids[None, :, :], axis=-1)
        _, t_match = linear_sum_assignment(cost)

        init_unk_weight = []
        for i in range(args.all_classes):
            if i not in t_match:
                init_unk_weight.append(t_centroids[i])
        init_unk_weight = np.stack(init_unk_weight, axis=0)

        for key, v in net.state_dict().items():
            if key == '1.main.1.2.weight':
                v.requires_grad = False
                net.state_dict()['1.fc.weight'].requires_grad = False

                vvnorm = (torch.norm(v, dim=-1)).mean().cpu().numpy()
                init_unk_weight = init_unk_weight / np.linalg.norm(init_unk_weight, axis=-1, keepdims=True) * vvnorm
                fcweight = np.concatenate([v[:args.shared_classes].clone().detach().cpu().numpy(),
                                          init_unk_weight], axis=0)
                param = torch.from_numpy(fcweight)
                if use_cuda:
                    param = param.cuda()
                net.state_dict()['1.fc.weight'].copy_(param)

                v.requires_grad = True
                net.state_dict()['1.fc.weight'].requires_grad = True

    # 更新GMM
    if epoch <= 30:
        gmm = BayesianGaussianMixture(n_components=4, max_iter=800).fit(ProbRecorder['kl'][:, None])
    else:
        gmm = BayesianGaussianMixture(n_components=2, max_iter=800).fit(ProbRecorder['kl'][:, None])
    if use_cuda:
        torch.cuda.empty_cache()

    # 评估
    with TrainingModeManager([feature_extractor, cls], train=False) as mgr, \
         Accumulator(['predict_prob','predict_index', 'label']) as accumulator:
        for (i, (im, label)) in enumerate(target_test):
            if use_cuda:
                im = im.cuda()
                label = label.cuda()
            ss, fs, _, predict_prob = net.forward(im)
            predict_prob, label = [variable_to_numpy(x) for x in (predict_prob, label)]
            label = np.argmax(label, axis=-1).reshape(-1, 1)
            predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)
            accumulator.updateData(globals())

    for x in list(accumulator.keys()):
        globals()[x] = accumulator[x]

    # 计算评估指标
    y_true = label.flatten()
    y_pred = predict_index.flatten()
    m = extended_confusion_matrix(
        y_true, y_pred,
        true_labels=(list(range(args.shared_classes)) + list(range(20, 31))),
        pred_labels=list(range(args.all_classes))
    )

    cm = m.astype(float) / np.sum(m, axis=1, keepdims=True)
    acc_os_star = sum([cm[i][i] for i in range(args.shared_classes)]) / args.shared_classes
    unkn = sum(sum([cm[i][args.shared_classes:] for i in range(10, 21)])) / 11
    acc_os = (acc_os_star * args.shared_classes + unkn) / 11
    hos = (2 * acc_os_star * unkn) / (acc_os_star + unkn) if (acc_os_star + unkn) > 0 else 0

    # 打印 epoch 结果
    ce_val = losscounter.ce / losscounter.batch
    entropy_val = losscounter.entropy / losscounter.batch
    virtual_val = losscounter.virtual / losscounter.batch
    ce_ep_val = losscounter.ce_ep / losscounter.batch
    adv_val = losscounter.adv / losscounter.batch

    print(f'Epoch:{epoch}\tOS: {acc_os:.3f}\tOS*:{acc_os_star:.3f}\tUnk:{unkn:.3f}\tHos:{hos:.3f}\t'
          f'ce: {ce_val:.3f}\tentropy:{entropy_val:.3f}\tvirtual:{virtual_val:.3f}\tce_ep:{ce_ep_val:.3f}\tadv:{adv_val:.3f}')

    # 更新最佳指标
    if hos > best_hos:
        best_os = acc_os
        best_os_star = acc_os_star
        best_unk = unkn
        best_hos = hos
        best_epoch = epoch

    if use_cuda:
        torch.cuda.empty_cache()

    epoch += 1

# 打印最佳结果（包含lambda参数）
print(f'\nBest (lambda1={args.lambda1}, lambda2={args.lambda2}, lambda3={args.lambda3}): '
      f'Epoch:{best_epoch}\tOS: {best_os:.3f}\tOS*:{best_os_star:.3f}\tUnk:{best_unk:.3f}\tHos:{best_hos:.3f}')
print(f'class_num: {args.all_classes}  {args}')

# 恢复标准输出
sys.stdout = sys.__stdout__
print(f'训练完成！日志已保存到: {log_file}')

# Colab环境下自动下载日志文件
try:
    from google.colab import files
    print('正在准备日志文件下载...')
    files.download(log_file)
except ImportError:
    pass
