import os
import mne

def get_file_name_with_number(dir,name):
    file = [n for n in os.listdir(dir) if name in n][0]
    return file

def ET2EEG_time(name):      #获取ET转EEG时间戳公式
    EEG_name = get_file_name_with_number('./EEG/edf/',name)   #获取EEG事件列表
    raw = mne.io.read_raw_edf('./EEG/edf/'+EEG_name)
    events_from_annot, event_dict = mne.events_from_annotations(raw)
    for i in range(0,len(events_from_annot)):
        if events_from_annot[i][2] == 3:
            EEG_marker = events_from_annot[i][0]
            #print(EEG_marker)
            break
        else:
            continue


    ET_name = get_file_name_with_number('./ET/object/',name)
    with open('./ET/object/'+ET_name, "r", encoding='gb18030', errors='ignore') as f:
        ET_data = [line.rstrip('\n').split(',') for line in f]
        for e in range(60, len(ET_data)):
            if ET_data[e][1] == 'MSG':
                Message = ET_data[e][3]
                if (Message[11:13]) == 'UE':
                    if Message[-1] == '3':
                        ET_marker=ET_data[e][0]
                        break
                    else:
                        continue
                else:
                    continue
            else:
                continue

    star_time = float(ET_marker)-float(EEG_marker)*2000
    return star_time

def blink2events(file_path1,file_path2,name,star_time):
    ET_name = get_file_name_with_number('./ET/object/',name)
    Chose_file =open(file_path1 + ET_name)
    data = [line.rstrip('\n').split(',') for line in Chose_file]
    file = open(file_path2 + name + '_blink.txt', 'a')
    for e in range(43,len(data)):
        #print(data[e][43]=='Blink')
        try:
            if (data[e][43] == 'Blink') and (data[e][44] =='Blink') :
                EEG_time1 = (float(data[e][0]) - star_time) / 2000
                EEG_time = EEG_time1 * 2 * 0.001+0.2
                if EEG_time<0:
                    continue
                points_info = [str(EEG_time), 'Blink']  # 持续时间，序号
                file.write('    '.join(points_info))
                file.write('\n')

        except:
            continue


if __name__ == '__main__':
    file_path1 = 'K:/object detection/ET/object/'
    file_path2 = 'K:/object detection/ET/'
    file_names = os.listdir(r'K:\object detection\ET\clip_old')
    star = []
    for name in file_names:
        #print(name)
        #name = '201911061117'
        star_time = ET2EEG_time(name)
        temp = [name,star_time]
        star.append(temp)

