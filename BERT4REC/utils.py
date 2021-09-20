import random 


def generate_random_mask(sample, mode, valid_sample_size, masking_rate, total_items, mask_index):
    """[입력 샘플을 마스킹한다.]
    Args:
        sample ([type]): [description]
        masking_rate ([type]): [description]
        total_items ([type]): [description]
        mask_index ([type]): [description]

    Returns:
        List[Int]: 마스킹 된 샘플 반환한다.
        List[Int]: 마스크 인덱스를 표기한다.
    """
    res = []
    mask = []
    origin_mask_rate = masking_rate # 80%
    random_item_mask_rate = origin_mask_rate + (1 - masking_rate) / (2.0) # 90%

    if mode == 'train':
        for item in sample:
            rate = random.random() # Random Number Generate
            
            if rate > random_item_mask_rate:
                res.append(random.randint(2, total_items))
                mask.append(True)
            elif rate > origin_mask_rate:
                res.append(mask_index)
                mask.append(True)
            else:
                res.append(item)
                mask.append(False)
    else:
        res = sample[:-valid_sample_size]
        mask = [False] * len(res)
        for item in sample[-valid_sample_size:]:
            if random.random() > .5:
                res.append(item)
                mask.append(False)
            else:
                res.append(mask_index)
                mask.append(True)
    return res, mask 

def pad_list(sample, mode, seq_len, pad_index):
    """[시퀀스 데이터에 패딩을 추가한다]

    Args:
        sample ([type]): [description]
        mode ([type]): [description]
        seq_len ([type]): [description]
        pad_index ([type]): [description]

    Returns:
        [List[int]]: [패딩된 시퀀스 데이터]
    """


    current_sample_size = len(sample)
    padding_count = seq_len - current_sample_size
    
    item_list = sample 
    if padding_count > 0:
        pads = [pad_index] * padding_count
        if mode == "left":
            item_list = pads + sample 
        else:
            item_list = sample + pads

    return item_list 


