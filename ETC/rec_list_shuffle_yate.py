import sys
import codecs

sys.stdin = codecs.getreader('utf-8')(sys.stdin.detach())
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

# Enter your code here. Read input from STDIN. Print output to STDOUT
"""
Fisher-Yates shuffle ( perfect random shuffling )
However, same artist playing two or three within a short time period
-> X, 음악의 아티스트 명을 참고해서 동일한 아티스트의 곱이 인접하지 않도록 셔플해야함
# spread out musics ( can check which artist's one)
# distance btw same artist's music
# using random offset
# shuffle( Fisher-Yates Shuffle ) the songs by the same artist among each other  ( recursively)
# main idea is very similar to the method used in dithering ( 간단히 점과 점을 일정 패턴으로 교차해서 찍으면서 색을 섞어 자연스럽게 )
# Floyd-Steinberg Dithering avoid clusters and produce much better results
# with the clusters sonf by the same songs 
# stretch songs along the total music playlist by artists

"""
import random 

def get_interval(input,total_num, playlist):
    intervals = []
    random_off = random.randrange(total_num)
    distance = int(total_num / len(input))
    intervals.append(random_off)
    
    for i in range(len(input)-1):
        temp = intervals[-1] + distance
        intervals.append(temp)
        
    
    if(intervals[-1] >= total_num):
        intervals[:] = [v - (intervals[-1] - total_num+1) for v in intervals]

    for interval, music in zip(intervals, input):
        #print(interval)
        if(playlist[interval] == ''):
            playlist[interval] = music
        else:
            playlist = playlist[:interval] + [music] + playlist[interval:]
    return playlist

T = int(input())
for t in range(T):
    titles = input().split('\t')
    artists = input().split('\t')
    total_songs = len(titles)
    m_info = dict()

    playlist = [''] * total_songs 

    for artist, title in zip(artists, titles):
        try:
            m_info[artist].append(title)
        except:
            m_info[artist] = [title,]
    
    for artist, musics in m_info.items():
        playlist = get_interval(musics, total_songs, playlist)
    

    playlist = '\t'.join(m for m in playlist if m!='')
    print(playlist)