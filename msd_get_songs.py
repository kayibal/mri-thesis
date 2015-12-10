
# coding: utf-8

# In[1]:

import os
import tables
import py7D
import glob
import sqlite3
import requests
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))


# In[ ]:

os.path.dirname(os.path.realpath("__file__"))


# In[2]:

os.chdir('code/msd_data/MillionSongSubset/data')


# In[ ]:

def create_db():
    conn = sqlite3.connect("processed_data.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE tracks (msd_id text, track_7digitalid integer, jam_id text, som_cluster integer, igmn_cluster integer, audio_path text, h5_path text, genre text)''')
    conn.commit()
    conn.close()


# In[3]:

def jam_mapping(path):
    mapping = {}
    with open(path) as f:
        for line in f:
            values = line.split('\t')
            mapping[values[1].strip('\n')] = values[0]
    return mapping


# In[4]:

jam_ids = jam_mapping('../../jam_to_msd.tsv')
#jam_ids


# In[ ]:

no_audio = []
no_jam = []
conn = sqlite3.connect("processed_data.db")
c = conn.cursor()

for(directory, _,files) in os.walk('.'):
    for h5 in glob.glob(os.path.join(directory,"*.h5")):
        print h5
        f = tables.open_file(h5, mode='r')
        track_id = f.root.analysis.songs.cols.track_id[0]
        audio_name = track_id +".mp3"
        audio_path = os.path.abspath(os.path.join(directory,audio_name))
        seven_id = f.root.metadata.songs.cols.track_7digitalid[0]
        #create db entry and get audiofile
        if not (os.path.exists(audio_path) and seven_id != -1):
            audio_url = py7D.preview_url(seven_id)
            r = requests.get(audio_url, headers={'Connection':'close'})
            if r.ok:
                #write audio file
                with open(audio_path, 'wb') as handle:
                    for block in r.iter_content(8*1024):
                        handle.write(block)
                #create db_entry
                query = "INSERT INTO tracks (msd_id, track_7digitalid, jam_id, audio_path, h5_path) VALUES ('%s',%d,'%s','%s','%s')"
                if jam_ids.has_key(track_id):
                    query = query % (track_id, seven_id, jam_ids[track_id], audio_path, os.path.abspath(h5))
                else:
                    query = query % (track_id, seven_id, -1, audio_path, h5)
                    no_jam.append(track_id)
                c.execute(query)
                conn.commit()
            else:
                no_audio.append(track_id)
            r.connection.close()
        elif seven_id == -1:
            no_audio.append(track_id)
        else:
            print "skip"
        f.close()
        
conn.close()
print "done"


# In[ ]:



