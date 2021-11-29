class Frame:
    def __init__(self):
        self.cheat = 0
        self.count = 0
        self.cheatcount = 0
        self.noface = 0
        self.multiface = 0
        self.facerec = 0
        self.head = 0
        self.mouth = 0
        self.spoof = 0
        self.emotion = 0
        self.framesleepy = 0
        self.sleepy = 0

class Segment:
    def __init__(self):
        self.cheat = 0
        self.count = 0
        self.cheatcount = 0

def detect_cheating_frame(faces,frames):

    frame=Frame()
    bool_flag=0
    if faces:
        if(len(faces)!=1):
            frame.multiface = 1
            bool_flag+=1

        for face in faces:
            if face.name=="Unknown":
                frame.facerec=1
                bool_flag+=1
            
            if face.spoof == False:
                frame.spoof = 1
                bool_flag+=1

            if face.head and face.head!="Head Straight":
                frame.head=1
                bool_flag+=1

            if face.mouth:
                if face.mouth.status=="mouth open":
                    frame.mouth=1
                    bool_flag+=1
            
            if face.emotion and face.emotion=="happy":
                frame.emotion=1
                bool_flag+=1
            
            if face.framesleepy and face.framesleepy>0:
                frame.framesleepy=1
                bool_flag+=1

            if face.sleepy and face.sleepy>0:
                frame.sleepy=1
                bool_flag+=1
    else:
        frame.noface = 1
        bool_flag+=1

    if(bool_flag>0):
        frame.cheat = 1
        frame.cheatcount = bool_flag
    else:
        frame.cheat = 0
        frame.cheatcount = 0

    frame.count=len(frames)+1
    return frame


def detect_cheating_segment(frames, fps):
#*************
    # Best values for the following parameters are chosen
    window_length = 5
    per = 0.2
    noface_cheat_threshold = 0.3
    multiface_cheat_threshold = 0.3
    facerec_cheat_threshold = 0.6
    head_cheat_threshold = 0.3
    mouth_cheat_threshold = 0.6
    spoof_cheat_threshold = 0.3
    emotion_cheat_threshold = 0.3
    sleepy_cheat_threshold = 0
#*************
    num_frames = len(frames)
    numframes_wd = window_length * fps
    
    # count variables
    segment_count=1

    segments = []       
    
    noface_per_frame = list(map(lambda x: x.noface , frames))
    multiface_per_frame = list(map(lambda x: x.multiface , frames))
    facerec_per_frame = list(map(lambda x: x.facerec , frames))
    head_per_frame = list(map(lambda x: x.head , frames))
    mouth_per_frame = list(map(lambda x: x.mouth , frames))
    spoof_per_frame = list(map(lambda x: x.spoof , frames))
    emotion_per_frame = list(map(lambda x: x.emotion , frames))
    sleepy_per_frame = list(map(lambda x: x.sleepy , frames))
    cheat_per_frame = list(map(lambda x: x.cheat , frames))

    for i in range(int(per*numframes_wd), num_frames, int(per*numframes_wd)):
        j = i+numframes_wd if i+numframes_wd < num_frames else num_frames-1
        
        noface_mean = sum(noface_per_frame[i:j+1]) / (j-i+1)
        multiface_mean = sum(multiface_per_frame[i:j+1]) / (j-i+1)
        facerec_mean = sum(facerec_per_frame[i:j+1]) / (j-i+1)
        head_mean = sum(head_per_frame[i:j+1]) / (j-i+1)
        mouth_mean = sum(mouth_per_frame[i:j+1]) / (j-i+1)
        spoof_mean = sum(spoof_per_frame[i:j+1]) / (j-i+1)
        emotion_mean = sum(emotion_per_frame[i:j+1]) / (j-i+1)
        sleepy_mean = sum(sleepy_per_frame[i:j+1]) / (j-i+1)
        
        seg = Segment()
        seg.count = segment_count
        segment_count+=1
        segments.append(seg)

        if((noface_mean>=noface_cheat_threshold) 
        or (multiface_mean>=multiface_cheat_threshold) 
        or (facerec_mean>=facerec_cheat_threshold) 
        or (head_mean>=head_cheat_threshold) 
        or (mouth_mean>=mouth_cheat_threshold) 
        or (spoof_mean>=spoof_cheat_threshold) 
        or (emotion_mean>=emotion_cheat_threshold) 
        or (sleepy_mean>=sleepy_cheat_threshold)):
            seg.cheat = 1
 
    return segments, cheat_per_frame

def detected_cheating(segments):
    detected = []
    for segment in segments:
        detected.append(segment.cheat)
    return detected


## FUNCTIONS RELATED TO ORIGINAL TRUTH VALUES: 

def truths_per_segment(data_path, sglen):
    wl = 5 #window_length
    per = 0.2
    fps,video_length,input_text = input_data(data_path)
    num_frames = fps*video_length
    numframes_wd = wl*fps
    frames = [0 for i in range(num_frames)] 
    original = []
    for i in range(len(input_text)):
        a=input_text[i][0]
        b=input_text[i][1]
        for j in range(a,b+1):
            frames[j] = 1
    for i in range(0, num_frames, int(per*numframes_wd)):
        j = i + numframes_wd if i+numframes_wd < num_frames else num_frames-1
        if sum(frames[i:j+1])/(j+1-i) >= 0.5 :
            original.append(1)
        else:
            original.append(0)
    return original

def time_to_frame(x,fps):
    minn = int(x[:2])
    sec = int(x[2:])
    no_frames = (minn*60+sec)*fps
    return no_frames

def input_data(data_path):
    data_file = open(data_path, "r")
    input_text = data_file. readlines()
    input_frwise_text = []
    fps,video_length = int(input_text[0].strip("\n")),int(input_text[1].strip("\n"))
    for ch_inst in input_text[2:]:
        start_f, end_f, tag = tuple(ch_inst.strip("\n").split(" "))
        start_f = time_to_frame(start_f, fps)
        end_f = time_to_frame(end_f,fps)
        input_frwise_text.append([start_f, end_f, tag])
    return fps,video_length,input_frwise_text

def get_accuracy(original,detected):
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    for i in range(min(len(original),len(detected))):
        if original[i]==0 and detected[i] ==0:
            tn+=1
        elif original[i]==1 and detected[i] ==0:
            fn+=1
        elif original[i]==0 and detected[i] ==1:
            fp+=1
        elif original[i]==1 and detected[i] ==1:
            tp+=1
       
    if tp+fp+tn+fn >0:
        accuracy = (tp+tn)/(tp+fp+fn+tn)
    else:
        accuracy = 0

    if tp+fp>0:
        precision = tp/(tp+fp)
    else:
        precision = 0

    if tp+fn>0:
        recall = tp/(tp+fn)
    else:
        recall = 0
        
    if precision+recall>0:
        f1score = 2*(precision*recall)/(precision+recall)
    else:
        f1score = 0

    print(f"Accuracy = {accuracy}\n Precision = {precision}\n Recall = {recall}\n F1Score = {f1score}")
    return accuracy,precision,recall,f1score
    