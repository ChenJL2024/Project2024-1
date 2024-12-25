import numpy as np  
  
# 初始化一个空列表来存储采样结果  
samples = []  
static_first = 0
# 目标采样次数  
target_samples = 10  
  
# 采样函数  
def sample_parameters():
    global static_first  
    # scale_x = np.random.uniform(0.5, 1.5)
    random_num = round(np.random.uniform(0.5, 1.5),2)

    translate_x = np.random.randint(-200, 201)  # 注意randint的结束边界是不包含的，所以加1  
    translate_y = np.random.randint(-200, 201)  
    static_first += 1
    return (random_num, translate_x, translate_y) , static_first
  
# 采样直到达到目标次数  
def get_samples():
    count = 0
    while len(samples) < target_samples:  
        count += 1
        new_sample,static_first = sample_parameters()  
        
        # 检查新采样是否与前面的采样至少有两个不同的值  
        is_valid = True  
        for prev_sample in samples:  
            if (new_sample[0] == prev_sample[0] or new_sample[1] == prev_sample[1] or new_sample[2] == prev_sample[2]):  
                is_valid = False  
                break  
        
        # 如果新采样是有效的（即与前面的采样至少有两个不同的值），则添加到列表中  
        if is_valid or static_first == 1:  
            samples.append(new_sample)
    print(count)
    return samples  

if __name__ == '__main__':
    samples = get_samples()
    # 打印采样结果  
    for idx, sample in enumerate(samples, 1):  
        print(f"Sample {idx}: scale_x={sample[0]}, translate_x={sample[1]}, translate_y={sample[2]}")