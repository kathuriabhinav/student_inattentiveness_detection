import numpy as np
import time 
import matplotlib.pyplot as plt

def print_list(items):
    for item in items:
        print(item,end=',')
    print()

def segment_list(items,segment_time,fps_assumed):
    no_of_frames = segment_time*fps_assumed
    count = 0
    listcount = []
    curr = 0
    for item in items:
        curr+=1
        if item == 1:
            count+=1

        if curr%no_of_frames == 0:
            listcount.append((count/no_of_frames)*100)
            count=0
    print_list(listcount)

def plot_main(frames,fps_assumed):
    segment_time = 1
    noface=[]
    multiface=[]
    facerec=[]
    head=[]
    mouth=[]
    spoof=[]
    emotion=[]
    sleepy=[]
    cheat=[]
    seg=[]
    no_of_frames = segment_time*fps_assumed
    for frame in frames:
        noface.append(frame.noface)
        multiface.append(frame.multiface)
        facerec.append(frame.facerec)
        head.append(frame.head)
        mouth.append(frame.mouth)
        spoof.append(frame.spoof)
        emotion.append(frame.emotion)
        sleepy.append(frame.framesleepy)
        cheat.append(frame.cheat)
        if(len(noface)%no_of_frames==0):
            seg.append(len(noface))
    cheat_list = [noface, multiface, facerec, head, mouth, spoof, emotion, sleepy, cheat]
    titles_list = ["No face Detected", 
    "Multiple faces Detected", 
    "User Authentication", 
    "Looking away from screen", 
    "Speaking", 
    "Spoofing Detected", 
    "Unnecessary Emotion Detected", 
    "Sleeping Detected", 
    "Estimated Inattentiveness Status"]
    
    plt.figure(figsize=(12,26))
    for i in range(9):
        plt.subplot(9,1,i+1)  
        plt.step(cheat_list[i],'r')
        plt.yticks([0,1]) 
        # plt.xticks(seg) 
        plt.title(titles_list[i])
        if(i==4):
            plt.ylabel("Inattentiveness status")
    plt.xlabel("frames")
    plt.subplots_adjust(hspace = 1.0)
    plt.savefig("results/Inattentiveness_frames_"+ time.strftime("%Y%m%d-%H%M%S") + ".png",dpi=300)
    plt.show()

    # print("Segment wise results")
    # for i in range(len(cheat_list)):
    #     print(titles_list[i])
    #     segment_list(cheat_list[i],segment_time,fps_assumed)
    return

def cheat_count(segments):
    cheat_count =0
    for segment in segments:
        if(segment.cheat):
            cheat_count+=1
    return cheat_count

def plot_segments(segments, original=[]):
    segment_time = 1
    x=[]
    detected=[]
    n = cheat_count(segments)
    for segment in segments:
        x.append(segment.count)
        detected.append(segment.cheat)

    plt.figure(figsize=(12,4)) 
    plt.step(detected,'r', linewidth = 3)
    plt.step(original,'b')
    plt.yticks([0,1])
    # plt.xticks(x)
    plt.xlabel('Time Segments')
    plt.ylabel('Inattentiveness')
    stats = "Total Time : " + str(len(segments)*segment_time) + " seconds\n" + "Inattentive for : " + str(n*segment_time) + " seconds"
    plt.figtext(0.5, 0.9,stats, ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":3})
    plt.savefig("results/Inattentiveness_detection_"+ time.strftime("%Y%m%d-%H%M%S") + ".png",dpi=300)
    plt.show()
    print(x)
    print(detected)