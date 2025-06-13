import os
import scipy.io as scio

def Save_mat(epoch , output_dim , datasets , query_labels , retrieval_labels , query_img , retrieval_img , save_dir='.' , mode_name="DSH",mAP=0):
    '''
    save_dir: 保存文件的目录路径
    output_dim: 输出维度
    datasets: 数据集名称
    query_labels: 查询图像的标签信息（numpy数组）
    retrieval_labels: 检索图像的标签信息（numpy数组）
    query_img: 查询图像的数据（numpy数组）
    retrieval_img: 检索图像的数据（numpy数组）
    mode_name: 模型的名称
    '''
    # print(query_labels)
    save_dir = os.path.join(save_dir , f'Hashcode_{datasets}_{output_dim}_{mode_name}')
    os.makedirs(save_dir,exist_ok=True)
    
    
    result_dict = {
        'q_img' : query_img ,
        'r_img' : retrieval_img ,
        'q_l' : query_labels ,
        'r_l' : retrieval_labels
    }

    filename = os.path.join(save_dir, f"{mAP}_{output_dim}-{epoch}-{datasets}-{mode_name}.mat")
    scio.savemat(filename, result_dict)

