import os  
import numpy as np  
import struct  
import PIL.Image  
import cfgs
   
train_data_dir = "HWDB1.1trn_gnt"  
test_data_dir = "HWDB1.1tst_gnt"  
chars = cfgs.classes
chars_np = np.zeros([len(chars),])
   
# 读取图像和对应的汉字  
def read_from_gnt_dir(gnt_dir=train_data_dir):  
    def one_file(f):  
        header_size = 10  
        while True:  
            header = np.fromfile(f, dtype='uint8', count=header_size)  
            if not header.size: break  
            sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)  
            tagcode = header[5] + (header[4]<<8)  
            width = header[6] + (header[7]<<8)  
            height = header[8] + (header[9]<<8)  
            if header_size + width*height != sample_size:  
                break  
            image = np.fromfile(f, dtype='uint8', count=width*height).reshape((height, width))  
            yield image, tagcode  
   
    for file_name in os.listdir(gnt_dir):  
        if file_name.endswith('.gnt'):  
            file_path = os.path.join(gnt_dir, file_name)  
            with open(file_path, 'rb') as f:  
                for image, tagcode in one_file(f):  
                    yield image, tagcode  
   

for image, tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):  
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')  
    """ 
    # 提取点图像, 看看什么样 
    """ 
    if tagcode_unicode in chars:
        label = chars.index(tagcode_unicode)
        im = PIL.Image.fromarray(image) 
        im.convert('RGB').save('png/' + str(int(chars_np[label])) + '_' + tagcode_unicode + '.png')
        chars_np[label] += 1 


