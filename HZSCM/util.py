# -*- coding: utf-8 -*-
"""
工具函数集合。

这里集中放了项目里最核心的辅助逻辑，包括：
1. 基于 CLIP 的伪标签生成
2. SAM / ERS 融合
3. patch 切分与整图拼接
4. 评价指标计算
5. 基于超像素聚类的伪标签修正

本文件里经常出现的变量约定：
- `data / image / HSI_image`：整图输入，通常形状为 `[H, W, C]`
- `gt / target / Train_Label`：真实标签图，通常形状为 `[H, W]`
- `sp_gt / SP_label / ERS_label`：超像素标签图，通常形状为 `[H, W]`
- `pseudo_probs / clip_probs`：伪标签概率图，通常形状为 `[H, W, num_classes]`
- `sp_features`：超像素级特征，形状为 `[num_superpixels, feature_dim]`
- `sp_labels`：超像素级标签，未标注超像素记为 `-1`
- `cluster_index`：每个超像素属于哪个聚类
"""
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from operator import truediv
from sklearn import metrics, neighbors
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sklearn.cluster as clu

try:
    import spectral as spy
except ImportError:
    spy = None
 

def get_pseudo_labels(SP_label, image, model, processor, label_name,  margin = 0, resize = True):
    """
    基于超像素区域，用 CLIP 为整幅图生成伪标签概率图。

    做法是：
    1. 遍历每个超像素
    2. 裁出该超像素对应的图像区域
    3. 用 CLIP 计算“这块区域属于每个类别文本”的概率
    4. 再把该概率回填给这个超像素内的所有像素

    参数说明：
    - `SP_label`：超像素编号图
    - `image`：RGB 图像
    - `label_name`：文本类别名列表
    - `margin`：裁剪时向外扩一圈的比例
    """
    device = model.device
    # `image_probs`：整幅图上每个像素的类别概率，最终返回的就是它。
    num_SP = np.max(SP_label + 1)
    image_probs = torch.zeros([image.shape[0], image.shape[1], len(label_name)])
    for i in np.unique(SP_label):
        i_row, i_col = np.where(SP_label == i)
        row_max = np.max(i_row)
        row_min = np.min(i_row)
        col_max = np.max(i_col)
        col_min = np.min(i_col) 
        row_length = int(margin * (row_max - row_min))
        col_length = int(margin * (col_max - col_min))
        # i_image = Image.fromarray(image[row_min:row_max+1, col_min:col_max+1])
        # margin 用来给裁剪框留一点边界上下文，但不能越界。
        if row_max + row_length >= image.shape[0] or row_min-row_length <= 0 or col_min-col_length <= 0 or col_max + row_length >= image.shape[1] :       
            i_image = Image.fromarray(image[row_min:row_max+1, col_min:col_max+1])            
        else:
            i_image = Image.fromarray(image[row_min - row_length:row_max + row_length+1, col_min - col_length:col_max+col_length+1])
        if resize :
            # CLIP 常用 224 输入尺寸，这里统一 resize。
            resized_image = i_image.resize((224, 224), Image.BICUBIC)
            inputs = processor(text= [f"a photo of {l}" for l in label_name], images=resized_image, return_tensors="pt", padding=True)
        else:
            inputs = processor(text= [f"a photo of {l}" for l in label_name], images=i_image, return_tensors="pt", padding=True)  
        for name, tensor in inputs.items():
            inputs[name] = tensor.to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1) 
        image_probs[i_row, i_col] = probs.cpu()    
    return image_probs  

def fuse_SAM_ERS(ERS_label, SAM_list):
    """
    把 SAM 的分割结果覆盖到 ERS 上，剩余区域仍保留 ERS 超像素。

    返回两个标签图：
    - `new_SAM_label`：SAM 覆盖到的区域编号
    - `new_ERS_label`：未被 SAM 覆盖的剩余 ERS 区域编号
    """
    SP_num = 0
    new_SAM_label = np.ones_like(ERS_label)*-100
    new_ERS_label = np.ones_like(ERS_label)*-100
    for i_list in SAM_list:
        i_SP = i_list['segmentation']
        new_SAM_label[i_SP] = SP_num
        SP_num = SP_num + 1
    residual_ERS_label = ERS_label[new_SAM_label<=-1]
    residual_ERS_index = np.unique(residual_ERS_label)
    
    for i in residual_ERS_index:
        new_ERS_label[ERS_label == i] = i
    return new_SAM_label, new_ERS_label
    
def get_pseudo_labels_using_SAM_and_ERS(ERS_label, SAM_list, image, model, processor, label_name, margin = 0, resize = True):
    """先融合 SAM 与 ERS，再基于融合结果生成伪标签。"""
    new_SAM_label, new_ERS_label = fuse_SAM_ERS(ERS_label, SAM_list)
    SAM_probs = get_pseudo_labels(new_SAM_label, image, model, processor, label_name, margin = margin, resize = resize)
    ERS_probs = get_pseudo_labels(new_ERS_label, image, model, processor, label_name, margin = margin, resize = resize)
    image_probs = torch.zeros_like(SAM_probs)
    image_probs[new_SAM_label >-1 ] = SAM_probs[new_SAM_label >-1]
    image_probs[new_SAM_label <=-1 ] = ERS_probs[new_SAM_label <=-1]
    return image_probs


def average_pseudo_probs(prob_list):
    """
    对多个概率图做平均，常用于多模态或多 CLIP 结果融合。

    `prob_list` 里的每一项都应该是同尺寸、同类别数的概率图。
    """
    if len(prob_list) == 0:
        raise ValueError('prob_list should not be empty')
    tensors = []
    for probs in prob_list:
        if torch.is_tensor(probs):
            tensors.append(probs.float())
        else:
            tensors.append(torch.as_tensor(probs, dtype=torch.float32))
    return torch.stack(tensors, dim=0).mean(dim=0)


def get_multimodal_pseudo_labels(SP_label, images, model, processor, label_name, margin = 0, resize = True):
    """分别为多个模态生成伪标签概率，再在概率层做平均。"""
    prob_list = [
        get_pseudo_labels(SP_label, image, model, processor, label_name, margin = margin, resize = resize)
        for image in images
    ]
    return average_pseudo_probs(prob_list)


def get_multimodal_pseudo_labels_using_SAM_and_ERS(ERS_label, SAM_list, images, model, processor, label_name, margin = 0, resize = True):
    """SAM+ERS 融合版本的多模态伪标签平均。只用高光谱的分割结果来生成伪标签，但输入图可以是多模态的。"""
    prob_list = [
        get_pseudo_labels_using_SAM_and_ERS(ERS_label, SAM_list, image, model, processor, label_name, margin = margin, resize = resize)
        for image in images
    ]
    return average_pseudo_probs(prob_list)


def get_multimodal_pseudo_labels_with_individual_segments(SP_labels, images, model, processor, label_name, margin = 0, resize = True):
    """
    多模态版本：每个模态使用自己的超像素分割结果生成伪标签，再在概率层平均。

    这和“多个模态共用同一份 SPGT”不同，这里要求 `SP_labels[i]` 与 `images[i]` 一一对应。
    """
    if len(SP_labels) != len(images):
        raise ValueError('SP_labels and images should have the same length')
    prob_list = [
        get_pseudo_labels(sp_label, image, model, processor, label_name, margin = margin, resize = resize)
        for sp_label, image in zip(SP_labels, images)
    ]
    return average_pseudo_probs(prob_list)


def get_multimodal_pseudo_labels_using_individual_SAM_and_ERS(ERS_labels, SAM_lists, images, model, processor, label_name, margin = 0, resize = True):
    """
    多模态版本：每个模态各自使用自己的 ERS 与 SAM 分割结果生成伪标签，再在概率层平均。

    参数中的三个列表必须一一对应：
    - `ERS_labels[i]`：第 i 个模态的 ERS 超像素图
    - `SAM_lists[i]`：第 i 个模态的 SAM mask 列表
    - `images[i]`：第 i 个模态转换得到的 RGB 图
    """
    if not (len(ERS_labels) == len(SAM_lists) == len(images)):
        raise ValueError('ERS_labels, SAM_lists, and images should have the same length')
    prob_list = [
        get_pseudo_labels_using_SAM_and_ERS(ers_label, sam_list, image, model, processor, label_name, margin = margin, resize = resize)
        for ers_label, sam_list, image in zip(ERS_labels, SAM_lists, images)
    ]
    return average_pseudo_probs(prob_list)

def accuracy(output, target, classcount):
    """旧版精度统计函数，当前主流程基本未使用。"""
    output=output.view(classcount,-1)
    target=target.view(1,-1)
    #
    m,n=output.size()
    _,L_output=torch.topk(output, 1, 0, True)
    #print(torch.max(L_output))
    #print(torch.nonzero(L_output).size(0))
    count=0
    aa=0
    for i in range(n):
        #print(L_target.data[0,i])
        if target[0,i]!=0  and L_output[0,i]==target[0,i]:
            aa=aa+1
        if target[0,i]!=0:
            count=count+1

    return aa, count

def ClassificationAccuracy(output, target, classcount):
    """
    计算每类精度、总体精度 OA 和平均精度 AA。

    变量说明：
    - `output`：预测标签图，标签编号通常为 `1..C`
    - `target`：真实标签图，背景为 0
    - `correct_perclass`：每类预测正确的像素数
    - `count_perclass`：每类真实像素总数
    """
    m, n = output.shape
    #output=np.reshape(output, [classcount, m*n])
    #target=np.reshape(target, [1, ])  #groundtruth label
    #m,n=output.size()
    #L_output=np.argmax(output, axis=0) # output label
    #_,L_output=torch.topk(output, 1, 0, True)

    correct_perclass=np.zeros([classcount])
    count_perclass = np.zeros([classcount])
    count=0
    aa=0

    for i in range(m):
        for j in range(n):
            if target[i, j]!=0:
                count=count+1
                count_perclass[int(target[i,j]-1)] += 1
                if output[i, j]==target[i, j]:
                    aa=aa+1
                    correct_perclass[int(target[i,j]-1)] += 1
            # if L_output[0,i]==7 or L_output[0,i]==9:
            #     print(target[0,i])
    test_AC_list = correct_perclass / count_perclass
    test_AA = np.average(test_AC_list)
    test_OA=aa/count

    return test_AC_list, test_OA, test_AA, aa, count


def Kappa(output, target, classcount):
    """计算 Cohen's kappa 和混淆矩阵。"""
    output=output
    target=target
    sizeOutput=np.shape(output)
    m=sizeOutput[0]
    n=sizeOutput[1]
    #output_data = np.transpose(output, (1,2,0))
    #idx = np.argmax(output, axis=0)

    test_pre_label_list = []
    test_real_label_list = []
    for ii in range(m):
        for jj in range(n):
            if target[ii][jj] != 0:
                test_pre_label_list.append(output[ii][jj])
                test_real_label_list.append(target[ii][jj])
    test_pre_label_list = np.array(test_pre_label_list)
    test_real_label_list = np.array(test_real_label_list)
    kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16), test_real_label_list.astype(np.int16))
    cm = metrics.confusion_matrix(test_real_label_list.astype(np.int16), test_pre_label_list.astype(np.int16))
    return kappa, cm

def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
    """使用 spectral 包把分类结果保存为彩色可视化图。"""
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    if spy is None:
        raise ImportError('spectral is required for Draw_Classification_Map')
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    pass


def SpiltHSI(data, gt, sp_gt, split_size, edge):
    """
    把整幅图切成多个 patch，并同步切分标签与伪标签。

    关键变量：
    - `split_height / split_width`：沿高和宽各切成多少块
    - `e`：每个 patch 额外保留的上下文边界宽度
    - `final_data / final_gt / final_spgt`：最终得到的 patch 列表
    """
    '''
    split HSI data with given slice_number
    :param data: 3D HSI data
    :param gt: 2D ground truth
    :param sp_gt: 2D ground truth
    :param split_size: [height_slice,width_slice]
    :return: splited data and corresponding gt
    '''
    e = edge  # 补边像素个数
    split_height = split_size[0]
    split_width = split_size[1]
    m, n, d = data.shape
    #gt = np.reshape(gt, [m, n])
    GT=gt
    SPGT = sp_gt
    #GT = GT_To_One_Hot(gt, class_count)

    # 将无法整除的块补0变为可整除
    if m % split_height != 0 or n % split_width != 0:
        data = np.pad(data, [[0, split_height - m % split_height], [0, split_width - n % split_width], [0, 0]],
                      mode='constant')
        GT = np.pad(GT, [[0, split_height - m % split_height], [0, split_width - n % split_width]],
                    mode='constant')
        SPGT = np.pad(SPGT, [[0, split_height - m % split_height], [0, split_width - n % split_width], [0, 0]],
                    mode='constant')
    m_height = int(data.shape[0] / split_height)
    m_width = int(data.shape[1] / split_width)

    pad_data = np.pad(data, [[e, e], [e, e], [0, 0]], mode="constant")
    pad_GT = np.pad(GT, [[e, e], [e, e]], mode="constant")
    pad_SPGT = np.pad(SPGT, [[e, e], [e, e],  [0, 0]], mode="constant")
    
    final_data = []
    final_gt=[]
    final_spgt=[]
    for i in range(split_height):
        for j in range(split_width):
            temp1 = pad_data[i * m_height:i * m_height + m_height + 2 * e, j * m_width:j * m_width + m_width + 2 * e, :]
            temp2 = pad_GT[i * m_height:i * m_height + m_height + 2 * e, j * m_width:j * m_width + m_width + 2 * e]
            temp3 = pad_SPGT[i * m_height:i * m_height + m_height + 2 * e, j * m_width:j * m_width + m_width + 2 * e]
            final_data.append(temp1)
            final_gt.append(temp2)
            final_spgt.append(temp3)
    final_data = np.array(final_data)
    final_gt = np.array(final_gt)
    final_spgt = np.array(final_spgt)

    return final_data, final_gt, final_spgt

def PatchStack(OutPut, m, n, patch_height, patch_width, split_height, split_width, EDGE, class_count):
    """
    把 patch 级预测重新拼回整图，并取 argmax 得到最终类别图。

    - `OutPut`：每个 patch 的类别 logits
    - `HSI_stack`：拼接后的整图 logits
    """

    HSI_stack = np.zeros([split_height * patch_height, split_width * patch_width, class_count], dtype=np.float32)
    for i in range(split_height):
        for j in range(split_width):
            if EDGE == 0:
                HSI_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :] = OutPut[
                                                                                                                   i * split_width + j][
                                                                                                               EDGE:,
                                                                                                               EDGE:,
                                                                                                               :]
            else:
                HSI_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :] = OutPut[
                                                                                                                   i * split_width + j][
                                                                                                               EDGE:-EDGE,
                                                                                                               EDGE:-EDGE, :]

    HSI_stack = np.argmax(HSI_stack, axis=2)

    # HSI_stack = HSI_stack[0: -(split_height - m % split_height), 0: -(split_width - n % split_width)]
    HSI_stack = HSI_stack[0:m , 0:n]
    return HSI_stack

def PatchStackFeat(OutPut, m, n, patch_height, patch_width, split_height, split_width, EDGE, class_count):
    """把 patch 级特征重新拼回整图。"""

    HSI_stack = np.zeros([split_height * patch_height, split_width * patch_width, OutPut[0].shape[-1]], dtype=np.float32)
    for i in range(split_height):
        for j in range(split_width):
            if EDGE == 0:
                HSI_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :] = OutPut[
                                                                                                                   i * split_width + j][
                                                                                                               EDGE:,
                                                                                                               EDGE:,
                                                                                                               :]
            else:
                HSI_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :] = OutPut[
                                                                                                                   i * split_width + j][
                                                                                                               EDGE:-EDGE,
                                                                                                               EDGE:-EDGE, :]
    # HSI_stack = np.argmax(HSI_stack, axis=2)
    HSI_stack = HSI_stack[0:m , 0:n]

    return HSI_stack

def PatchStackLabel(OutPut, m, n, patch_height, patch_width, split_height, split_width, EDGE, class_count):
    """把 patch 级标签重新拼回整图。"""
    
    HSI_stack = np.zeros([split_height * patch_height, split_width * patch_width], dtype=np.int32)
    for i in range(split_height):
        for j in range(split_width):
            if EDGE == 0:
                HSI_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width] = OutPut[
                                                                                                                   i * split_width + j][
                                                                                                               EDGE:,
                                                                                                               EDGE:]
            else:
                HSI_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width] = OutPut[
                                                                                                                   i * split_width + j][
                                                                                                               EDGE:-EDGE,
                                                                                                               EDGE:-EDGE]
    HSI_stack = HSI_stack[0:m , 0:n]

    return HSI_stack
    

def split_image(image, n):
    """把图像继续均匀切成更小的 n×n 子块。"""
    image = image.transpose(0,3,1,2)
    b, c, h, w = image.shape
    pad_h = n - h % n
    pad_w = n - w % n
    image = np.pad(image, [[0,0], [0,0], [0, pad_h], [0, pad_w]], mode='constant')
    b, c, h, w = image.shape
    image = image.reshape(b, c, h // n, n, w // n, n)
    image = image.transpose(0, 2, 4, 1, 3, 5)
    image = image.reshape(-1, c, n, n)
    return image, (h, w)

def get_USH(ac, test_num, unseen_classes, metric = 'OA'):
    """
    计算 seen / unseen / harmonic 三个零样本评价指标。

    - `ac`：每类精度
    - `test_num`：每类测试样本数
    - `unseen_classes`：被留出的 unseen 类别编号列表（从 0 开始）
    """
    # seen_index = [i if i not in unseen_classes for i in range(len(ac))]
    if unseen_classes == None:
        return 0, 0, 0
    else:
        test_num = np.array(test_num)
        seen_classes= [i for i in range(len(ac)) if i not in unseen_classes]
        if metric == 'OA':
            S = np.sum(ac[seen_classes] * test_num[seen_classes])/np.sum(test_num[seen_classes])
            U = np.sum(ac[unseen_classes] * test_num[unseen_classes])/np.sum(test_num[unseen_classes])
        elif metric == 'AA':
            S = np.mean(ac[seen_classes])
            U = np.mean(ac[unseen_classes])
        else:
            raise Exception('metric should be chosen from [OA, AA]')   
        H = 2 * S * U / (S + U)    
        return S, U, H 
               


def local_aggregation(unseen_classes, sp_features, sp_labels, cluster_index, pseudo_probs, threshold, tao):
    """
    基于“局部一致性”修正超像素级伪标签。

    核心思想：
    1. 先根据超像素特征计算相似度
    2. 再看某个超像素当前伪标签，是否和它所在聚类中的同类邻居足够一致
    3. 如果一致性太差，就尝试把该超像素改成次高概率类别

    关键变量：
    - `sim`：超像素两两之间的相似度矩阵
    - `pseudo_labels`：当前超像素伪标签
    - `pseudo_probs_c`：可能会被交换类别概率位置的修正版概率
    - `scores`：每个超像素在不同候选类别上的局部一致性分数
    """
    sim = - metrics.pairwise_distances(sp_features, metric='euclidean') / tao   
    pseudo_labels = np.argmax(pseudo_probs, axis = 1)
    pseudo_probs_c = np.copy(pseudo_probs)
    
    # neigh = neighbors.NearestNeighbors(n_neighbors=11, radius=0.4)
    # neigh.fit(sp_feature)
    
    ind = sp_labels > -0.5
    # pseudo_labels[ind] = sp_labels[ind]
    pseudo_labels_1 = np.copy(pseudo_labels)

    num_fea = sp_features.shape[0]
    scores = np.ones_like(pseudo_probs) * -100
    for i in range(num_fea):
        j = 0
        while pseudo_labels[i] not in unseen_classes:
            # print('pseudo_labels:', pseudo_labels[i])
            # prob = np.exp(sim[i]) / (np.sum(np.exp(sim[i]))-1)
            # 把距离转成相似度分布，作为邻居加权系数。
            prob = np.exp(sim[i]) / np.sum(np.exp(sim[i]))
            # rank = np.argsort(prob)

            ps_label = pseudo_labels[i]
            cs_label = cluster_index[i]
            
            spec_neigh_ind = cluster_index == cs_label
            # spec_neigh_ind = rank >= (prob.shape[0] - 100 -1)
            spec_neigh_ind[i] = False
            
            class_neigh_ind = sp_labels == ps_label
            # class_neigh_ind = pseudo_labels_1 == ps_label
            class_neigh_ind[i] = False
            intersection = spec_neigh_ind & class_neigh_ind
            
            cs_prob = np.sum(prob[intersection])
            class_prob = np.sum(prob[class_neigh_ind])+1e-10
            # class_prob = np.sum(prob[spec_neigh_ind])+1e-6
            # score 越大，说明“这个标签在局部邻域里越自洽”。
            score = cs_prob/class_prob
            scores[i, j] = score
            if score >= 1:
                print('num_fea:', i, 'cs_prob:', cs_prob, 'class_prob:', class_prob)
            # 一旦低于阈值，就把当前类别替换成下一个候选类别继续检查。
            if score < threshold:
                j = j + 1               
                m = np.copy(pseudo_labels[i])   #原位置
                pseudo_labels[i] = np.argsort(pseudo_probs_c[i])[-j-1]   #现位置
                n = np.copy(pseudo_probs_c[i][pseudo_labels[i]])  #现位置上的值
                pseudo_probs_c[i][pseudo_labels[i]] =  pseudo_probs_c[i][m]#现位置上赋上原位的值
                pseudo_probs_c[i][m] = n  #原位置上赋上现位的值                              
            else:
                break
    assert np.all(np.argmax(pseudo_probs_c, axis=1) == pseudo_labels)
    return scores, pseudo_labels, pseudo_probs_c

# def get_proper_threshold(unseen_classes, sp_features, sp_labels, cluster_index, pseudo_labels, tao):
#     sim = - metrics.pairwise_distances(sp_features, metric='euclidean') / tao
#     num_fea = sp_features.shape[0]
#     scores = np.ones_like(sp_features[:,0]) * -100
#     for i in range(num_fea):
#         if pseudo_labels[i] not in unseen_classes:
#             prob = np.exp(sim[i]) / (np.sum(np.exp(sim[i]))-1)
#             ps_label = pseudo_labels[i]
#             cs_label = cluster_index[i]
#             spec_neigh_ind = cluster_index == cs_label
#             spec_neigh_ind[i] = False

#             class_neigh_ind = sp_labels == ps_label
#             class_neigh_ind[i] = False
#             intersection = spec_neigh_ind & class_neigh_ind

#             cs_prob = np.sum(prob[intersection])
#             class_prob = np.sum(prob[class_neigh_ind])+ 1e-10
#             score = cs_prob / class_prob  # can use log to get negative score
#             scores[i] = score
#     return scores, pseudo_labels

def get_sp_label(HSI_image, sp_gt, train_gt):
    """
    把像素级数据聚合到超像素级。

    返回：
    - 每个超像素的平均特征
    - 每个超像素的监督标签（如果该超像素没有已知标签，则记为 -1）
    - 超像素编号列表
    """
    # `sp`：超像素编号列表。
    sp = np.unique(sp_gt)
    sp_features = []
    sp_labels = []
    for i in sp:
        sp_ind = np.where(sp_gt == i)
        sp_label = train_gt[sp_ind]
        count = np.bincount(sp_label)[1:]
        if np.sum(count) > 0.5:
            sp_label = np.argmax(count)
        else:
            sp_label = -1
        sp_labels.append(sp_label)
        sp_features.append(np.mean(HSI_image[sp_ind], axis=0))
    sp_features = np.array(sp_features)
    sp_labels = np.array(sp_labels)
    return sp_features, sp_labels, sp

def get_cluster_labels(sp_features, sp_labels, n_cluster=40):
    """
    对超像素特征做谱聚类，并把簇内多数标签传播给未标注超像素。
    """
    # 这里选用谱聚类，而不是普通 k-means。
    kmeans = clu.SpectralClustering(n_clusters=n_cluster, assign_labels='cluster_qr').fit(sp_features)
    cluster_index = kmeans.labels_
    clusters = []
    for i in np.unique(cluster_index):
        clusters.append(np.where(cluster_index == i)[0])
    p_label = np.copy(sp_labels)
    for i in range(len(clusters)):
        cla = np.zeros(np.max(sp_labels)+1, dtype = np.int32)
        for j in clusters[i]:
            ind = j
            if sp_labels[ind] >= 0:
                cla[sp_labels[ind]] += 1
        if np.sum(cla) > 0:
            for j in clusters[i]:
                ind = j
                if sp_labels[ind] < 0:
                    p_label[ind] = np.argmax(cla)        
    return clusters, cluster_index, p_label

def get_whole_labels(clip_ps_labels, ps_labels, sp_gt):
    """
    把超像素级标签重新回填成像素级整图标签。

    `clip_ps_labels` 主要用于提供输出数组的形状；真正的标签值由 `ps_labels` 覆盖进去。
    """
    sp = np.unique(sp_gt)
    whole_labels = np.copy(clip_ps_labels)
    for i in sp:
        sp_ind = np.where(sp_gt == i)
        whole_labels[sp_ind] = ps_labels[i]
    return whole_labels



def correct_pseudo_labels(HSI_image, gt, sp_gt, train_gt, clip_probs, net_ps_labels, unseen_classes, n_neighbours=10, n_clusters=40, tao=1.0, threshold=0.1):
    """
    伪标签修正总入口。

    整个流程是：
    1. 先把 CLIP 概率和网络预测都聚合到超像素级
    2. 再对超像素做聚类
    3. 利用局部一致性修正超像素伪标签
    4. 最后把已知监督标签强制覆盖回去，避免真值被改坏

    关键变量：
    - `clip_ps_probs`：CLIP 概率聚合到超像素后的结果
    - `net_ps_labels`：网络整图预测再压到超像素后的多数标签
    - `sp_features / sp_labels`：超像素级特征与监督标签
    - `whole_pseudo_labels / whole_pseudo_probs`：最终回填到整图后的伪标签结果
    """    
    clip_ps_probs,_,_ = get_sp_label(clip_probs, sp_gt, train_gt.astype(np.int32))
    # `net_ps_labels` 在当前版本里主要作为辅助分析保留，并没有深度参与最终决策。
    net_ps_labels = np.array([np.argmax(np.bincount(net_ps_labels[np.where(sp_gt==i)])) for i in np.unique(sp_gt)])
    
    sp_features, sp_labels, sp_index = get_sp_label(HSI_image, sp_gt, train_gt.astype(np.int32))

    # clusters, cluster_labels = get_aroc_label(sp_features, sp_labels, n_neighbours, threshold) # 超像素块的聚类集合  以及每个超像素块的聚类标签， -1表示有些聚类中的超像素块没有分配标签
    # cluster_index = assign_cluster_labels(clusters)  #每个超像素块的聚类索引
    clusters, cluster_index, cluster_labels = get_cluster_labels(sp_features, sp_labels, n_cluster=n_clusters)


    scores, pseudo_labels, pseudo_prob = local_aggregation(unseen_classes, sp_features, sp_labels, cluster_index, clip_ps_probs, threshold, tao)

    # 已知监督标签的超像素直接恢复成真值，不再依赖伪标签结果。
    ind = sp_labels > -0.5
    pseudo_labels[ind] = sp_labels[ind]
    pseudo_prob[ind] = np.eye(np.max(gt))[sp_labels[ind]]

    whole_pseudo_labels = get_whole_labels(np.argmax(clip_probs, axis=-1),pseudo_labels, sp_gt)
    whole_pseudo_probs = get_whole_labels(clip_probs,pseudo_prob, sp_gt)

    # ce_label=np.argmax(clip_ps_probs, axis = -1)
    # ce_label[ind]=net_ps_labels[ind]

    # t_scores, _ = get_proper_threshold(unseen_classes, sp_features, sp_labels, cluster_index, ce_label, tao)
    # aa = t_scores[ind]
    assert np.all(np.argmax(whole_pseudo_probs, axis=-1) == whole_pseudo_labels)
    return scores, whole_pseudo_labels, whole_pseudo_probs
    
    
"""


aaa = t_scores[sp_labels>-0.5]

for i in np.unique(sp_labels):
    print('labels:',i, 'num:', np.sum(sp_labels == i))
    
for i in np.unique(cluster_labels):
    print('labels:',i, 'num:', np.sum(cluster_labels == i))
"""
        


class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, constant_predictor_lr=False):
        # 这是一个“先 warmup，再 cosine 衰减”的学习率调度器。
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        #warmup_iter = iter_per_epoch * warmup_epochs
        warmup_iter = warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        #decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        decay_iter = num_epochs - warmup_epochs
        cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0
    def step(self):
        # 每调用一次，就把优化器学习率推进到下一个时间点。
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr
    def get_lr(self):
        return self.current_lr


        
    
    
