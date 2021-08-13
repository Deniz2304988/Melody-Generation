import os, sys, stat
import json
import tensorflow.keras as keras
import music21 as mu
import numpy as np
env = mu.environment.Environment(forcePlatform='darwin')
env['musescoreDirectPNGPath'] = r'C:/Program Files/MuseScore 3'
env['musicxmlPath'] = r'C:/Program Files/MuseScore 3'

Sequence_length=64

Duration_Acceptable= [
    0.25,
    0.5,
    0.75,
    1,
    1.5,
    1.75,
    2,
    3,
    4

]
def duration_check(song,duration_acceptable):
    for note in song.notesAndRest:
        if note.duration.quarterlength not in duration_acceptable:
            return False
        else:
            return True

    pass


Kern_Path="C:/Users/user/Desktop/data/deutschl/test"
def load_songs(data_path):
    # go through all the files and load them with music 21
    songs=[]
    for path,subdirs,files in os.walk(data_path):
        for file in files:
            if file[-4:] == ".krn":
                added_song=mu.converter.parse(os.path.join(path,file))
                songs.append(added_song)


    return songs



def encoding_func(song,time_step=0.25):
    encoded_song=[]
    for event in song.flat.notesAndRests:

        if isinstance(event,mu.note.Note):
            symbol=event.pitch.midi
        if isinstance(event,mu.note.Rest):
            symbol="r"
        steps = int(event.duration.quarterLength / time_step)

        for step in range(steps):
            if step==0:
                encoded_song.append(symbol)

            else:
                encoded_song.append("-")


    encoded_song=" ".join(map(str,encoded_song))
    return encoded_song


def preprocess(data_path):
    mappings={}
    # load songs
    songs=load_songs(data_path)
    fp=songs[0].write("midi","song1.mid")

    for j,song in enumerate(songs):
        encoded_song=encoding_func(song,0.25)

        save_path=os.path.join("C:/Users/user/PycharmProjects/Melody_Generation/new_dataset",str(j))
        with open(save_path,"w") as fp:
            fp.write(encoded_song)


    dataset_create("C:/Users/user/PycharmProjects/Melody_Generation/new_dataset/dataset","C:/Users/user/PycharmProjects/Melody_Generation/new_dataset",64)
    with open("C:/Users/user/PycharmProjects/Melody_Generation/new_dataset/dataset", "r") as fp:
        songs = fp.read()

    songs=songs.split()
    vocab=list(set(songs))

    for i,symbol in enumerate(vocab):
        mappings[symbol]=i

    with open("C:/Users/user/PycharmProjects/Melody_Generation/new_dataset\mapping","w") as fp:
        json.dump(mappings,fp,indent=4)

    inputs,targets=create_melody_sequence(songs,Sequence_length)

    with open('input.npy', 'wb') as f:
        np.save(f, inputs)

    with open('targets.npy', 'wb') as f:
        np.save(f, targets)








    # save songs to text file
def dataset_create(dataset_path,file_path,sequence_length):
    delimeter="/ " * sequence_length
    songs=""
    for path, _, files in os.walk(file_path):

        for file in files:
            created_path=os.path.join(path,file)
            with open(created_path,"r") as fp:
                song=fp.read()



            songs=songs + song + " " + delimeter

    songs=songs[:-1]

    with open(dataset_path, "w") as fp:
        fp.write(songs)

    return songs

def create_melody_sequence(songs,sequence_length):
    int_songs=[]

    with open("C:/Users/user/PycharmProjects/Melody_Generation/new_dataset/mapping","r") as fp:
        mapping=json.load(fp)

    for symbol in songs:
        int_songs.append(mapping[symbol])

    inputs=[]
    targets=[]
    num_sequences=len(int_songs) - sequence_length

    for k in range(num_sequences):
        inputs.append(int_songs[k:k+sequence_length])
        targets.append(int_songs[k+sequence_length])


    vocab_size=len(set(int_songs))

    inputs=keras.utils.to_categorical(inputs,num_classes=vocab_size)
    targets=np.array(targets)

    return inputs,targets












preprocess(Kern_Path)