from tkinter import *
import os
os.getcwd()
import pygame
from tkinter import filedialog
root = Tk()
root.title('Musicplayer App')

# Initialize Pygame Mixer
pygame.mixer.init()

# Add Song Function
def add_song():
    song=filedialog.askopenfilename(initialdir='/archive )',title='Choose a song', filetypes=(("wav Files","*.wav"),))

    # strip out the directory info and.wav extension from the song
    song=song.replace("C:/Users/asmin/PycharmProjects/EBMP/archive","")
    song =song.replace(".wav","")
    # Add song to the listbox
    song_box.insert(END, song)

# Add Many Songs To Playlist
def add_many_songs():
    songs = filedialog.askopenfilenames(initialdir='/archive ', title='Choose a song',
                                      filetypes=(("wav Files", "*.wav"),))
    # loop through song lists and replace directory info and wav
    for song in songs:
        song = song.replace("C:/Users/asmin/PycharmProjects/EBMP/archive", "")
        song = song.replace(".wav", "")
        # Add song to the listbox
        song_box.insert(END, song)



#     play selected song
def play():
    song=song_box.get(ACTIVE)
    song=f'C:/Users/asmin/PycharmProjects/EBMP/archive/{song}.wav'
    pygame.mixer.music.load(song)
    pygame.mixer.music.play(loops=0)
# stop playing current song
def stop():
    pygame.mixer.music.stop()
    song_box.selection_clear(ACTIVE)

# Play the next song in the playlist
def next_song():
    # Get the current song tuple number
    next_one = song_box.curselection()
    # Add one to the current song number
    next_one = next_one[0]+1
    # Grab Song title from playlist
    song = song_box.get(next_one)
    # Add directory structure and wav to song title
    song = f'C:/Users/asmin/PycharmProjects/EBMP/archive/{song}.wav'
    # load and play song
    pygame.mixer.music.load(song)
    pygame.mixer.music.play(loops=0)

    # clear active bar in playlist listbox
    song_box.selection_clear(0,END)

    # Activate new song bar
    song_box.activate(next_one)

    # set active bar to next song
    song_box.selection_set(next_one,last=None)

# play previous song in playlist
def previous_song():
    # Get the current song tuple number
    next_one = song_box.curselection()
    # Add one to the current song number
    next_one = next_one[0] - 1
    # Grab Song title from playlist
    song = song_box.get(next_one)
    # Add directory structure and wav to song title
    song = f'C:/Users/asmin/PycharmProjects/EBMP/archive/{song}.wav'
    # load and play song
    pygame.mixer.music.load(song)
    pygame.mixer.music.play(loops=0)

    # clear active bar in playlist listbox
    song_box.selection_clear(0, END)

    # Activate new song bar
    song_box.activate(next_one)

    # set active bar to next song
    song_box.selection_set(next_one, last=None)

# create Global paused variable
global paused
paused = False
#Pause and unpause the current song
def pause(is_paused):
    global paused
    paused = is_paused

    if paused:
        # Unpause
        pygame.mixer.music.unpause()
        paused=False
    else:
        # pause
        pygame.mixer.music.pause()
        paused=True

# delete a song
def delete_song():
    # delete currently selected song
    song_box.delete(ANCHOR)
    pygame.mixer.music.stop()

# delete all songs from playlist
def delete_all_songs():
    # delete all songs
    song_box.delete(0,END)
    pygame.mixer.music.stop()

# Create playlist Box
song_box = Listbox(root, bg="black",fg="green",width=80 ,height= 20, selectbackground="gray",selectforeground="blue")
song_box.pack(pady=30)

# Define player control buttons images
back_btn_img= PhotoImage(file='photos/back50.png')
forward_btn_img= PhotoImage(file='photos/forward_button50.png')
play_btn_img = PhotoImage(file='photos/play50.png')
pause_btn_img = PhotoImage(file='photos/pause50.png')
stop_btn_img = PhotoImage(file='photos/stop50.png')

# Create player control frame
controls_frame = Frame(root)
controls_frame.pack()

# Create Player Control Buttons
back_button = Button(controls_frame, image=back_btn_img, borderwidth=0,command=previous_song)
forward_button = Button(controls_frame, image=forward_btn_img, borderwidth=0,command=next_song)
play_button = Button(controls_frame, image=play_btn_img, borderwidth=0,command=play)
pause_button = Button(controls_frame, image=pause_btn_img, borderwidth=0,command=lambda:pause(paused))
stop_button = Button(controls_frame, image=stop_btn_img, borderwidth=0,command=stop)

back_button.grid(row=0,column=0,padx=10)
forward_button.grid(row=0,column=1,padx=10)
play_button.grid(row=0,column=2,padx=10)
pause_button.grid(row=0,column=3,padx=10)
stop_button.grid(row=0,column=4,padx=10)

# Create Menu
my_menu = Menu(root)
root.config(menu=my_menu)

# Create Add Song Menu
add_song_menu = Menu(my_menu)
my_menu.add_cascade(label="Add Songs", menu= add_song_menu)
add_song_menu.add_command(label="Add One Song To Playlist", command=add_song)

# Create Add  Many Songs to playlist
add_song_menu.add_command(label="Add Many Songs To Playlist", command=add_many_songs)

# Create Delete Song menu
remove_song_menu = Menu(my_menu)
my_menu.add_cascade(label="Remove Songs", menu= remove_song_menu)
remove_song_menu.add_command(label="Delete a Song From Playlist", command=delete_song)
remove_song_menu.add_command(label="Delete all Songs From Playlist", command=delete_all_songs)





root.mainloop()