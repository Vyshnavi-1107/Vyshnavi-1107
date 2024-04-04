
(with case study)
JalFaizy Shaikh
JalFaizy Shaikh
27 Aug, 2021 • 8 min read
Introduction
When you get started with data science, you start simple. You go through simple projects like Loan Prediction problem or Big Mart Sales Prediction. These problems have structured data arranged neatly in a tabular format. In other words, you are spoon-fed the hardest part in data science pipeline.



The datasets in real life are much more complex.

You first have to understand it, collect it from various sources and arrange it in a format which is ready for processing. This is even more difficult when the data is in an unstructured format such as image or audio. This is so because you would have to represent image/audio data in a standard way for it to be useful for analysis.

 

The abundance on unstructured data
Interestingly, unstructured data represents huge under-exploited opportunity. It is closer to how we communicate and interact as humans. It also contains a lot of useful & powerful information. For example, if a person speaks; you not only get what he / she says but also what were the emotions of the person from the voice.


Also the body language of the person can show you many more features about a person, because actions speak louder than words! So in short, unstructured data is complex but processing it can reap easy rewards.

In this article, I intend to cover an overview of audio / voice processing with a case study so that you would get a hands-on introduction to solving audio processing problems.

Let’s get on with it!

 

Table of Contents
What do you mean by Audio data?
Applications of Audio Processing
Data Handling in Audio domain
Let’s solve the UrbanSound challenge!
Intermission: Our first submission
Let’s solve the challenge! Part 2: Building better models
Future Steps to explore
 

What do you mean by Audio data?
Directly or indirectly, you are always in contact with audio. Your brain is continuously processing and understanding audio data and giving you information about the environment. A simple example can be your conversations with people which you do daily. This speech is discerned by the other person to carry on the discussions. Even when you think you are in a quiet environment, you tend to catch much more subtle sounds, like the rustling of leaves or the splatter of rain. This is the extent of your connection with audio.

So can you somehow catch this audio floating all around you to do something constructive? Yes, of course! There are devices built which help you catch these sounds and represent it in computer readable format. Examples of these formats are

wav (Waveform Audio File) format
mp3 (MPEG-1 Audio Layer 3) format
WMA (Windows Media Audio) format
If you give a thought on what an audio looks like, it is nothing but a wave like format of data, where the amplitude of audio change with respect to time. This can be pictorial represented as follows.

Applications of Audio Processing
Although we discussed that audio data can be useful for analysis. But what are the potential applications of audio processing? Here I would list a few of them

Indexing music collections according to their audio features.
Recommending music for radio channels
Similarity search for audio files (aka Shazam)
Speech processing and synthesis – generating artificial voice for conversational agents
Here’s an exercise for you; can you think of an application of audio processing that can potentially help thousands of lives?

 

Data Handling in Audio domain
As with all unstructured data formats, audio data has a couple of preprocessing steps which have to be followed before it is presented for analysis.. We will cover this in detail in later article, here we will get an intuition on why this is done.

The first step is to actually load the data into a machine understandable format. For this, we simply take values after every specific time steps. For example; in a 2 second audio file, we extract values at half a second. This is called sampling of audio data, and the rate at which it is sampled is called the sampling rate.



Another way of representing audio data is by converting it into a different domain of data representation, namely the frequency domain. When we sample an audio data, we require much more data points to represent the whole data and also, the sampling rate should be as high as possible.

On the other hand, if we represent audio data in frequency domain, much less computational space is required. To get an intuition, take a look at the image below



Source

Here, we separate one audio signal into 3 different pure signals, which can now be represented as three unique values in frequency domain.

There are a few more ways in which audio data can be represented, for example. using MFCs (Mel-Frequency cepstrums. PS: We will cover this in the later article). These are nothing but different ways to represent the data.

Now the next step is to extract features from this audio representations, so that our algorithm can work on these features and perform the task it is designed for. Here’s a visual representation of the categories of audio features that can be extracted.



After extracting these features, it is then sent to the machine learning model for further analysis.

 

Let’s solve the UrbanSound challenge!
Let us have a better practical overview in a real life project, the Urban Sound challenge. This practice problem is meant to introduce you to audio processing in the usual classification scenario.

The dataset contains 8732 sound excerpts (<=4s) of urban sounds from 10 classes, namely:

air conditioner,
car horn,
children playing,
dog bark,
drilling,
engine idling,
gun shot,
jackhammer,
siren, and
street music
Here’s a sound excerpt from the dataset. Can you guess which class does it belong to?

Audio Player

00:00
00:00
To play this in the jupyter notebook, you can simply follow along with the code.

import IPython.display as ipd
ipd.Audio('../data/Train/2022.wav')
Now let us load this audio in our notebook as a numpy array. For this, we will use librosa library in python. To install librosa, just type this in command line

pip install librosa
Now we can run the following code to load the data

data, sampling_rate = librosa.load('../data/Train/2022.wav')
When you load the data, it gives you two objects; a numpy array of an audio file and the corresponding sampling rate by which it was extracted. Now to represent this as a waveform (which it originally is), use the following  code

% pylab inline
import os
import pandas as pd
import librosa
import glob 

plt.figure(figsize=(12, 4))
librosa.display.waveplot(data, sr=sampling_rate)
The output comes out as follows



Let us now visually inspect our data and see if we can find patterns in the data

Class:  jackhammer


Class: drilling

Class: dog_barking

We can see that it may be difficult to differentiate between jackhammer and drilling, but it is still easy to discern between dog_barking and drilling. To see more such examples, you can use this code

i = random.choice(train.index)

audio_name = train.ID[i]
path = os.path.join(data_dir, 'Train', str(audio_name) + '.wav')

print('Class: ', train.Class[i])
x, sr = librosa.load('../data/Train/' + str(train.ID[i]) + '.wav')

plt.figure(figsize=(12, 4))
librosa.display.waveplot(x, sr=sr)
 

 

Intermission: Our first submission
We will do a similar approach as we did for Age detection problem, to see the class distributions and just predict the max occurrence of all test cases as that class.

Let us see the distributions for this problem.

train.Class.value_counts()
Out[10]:

jackhammer 0.122907
engine_idling 0.114811
siren 0.111684
dog_bark 0.110396
air_conditioner 0.110396
children_playing 0.110396
street_music 0.110396
drilling 0.110396
car_horn 0.056302
gun_shot 0.042318
We see that jackhammer class has more values than any other class. So let us create our first submission with this idea.

test = pd.read_csv('../data/test.csv')
test['Class'] = 'jackhammer'
test.to_csv(‘sub01.csv’, index=False)
This seems like a good idea as a benchmark for any challenge, but for this problem, it seems a bit unfair. This is so because the dataset is not much imbalanced.

 

Let’s solve the challenge! Part 2: Building better models
Now let us see how we can leverage the concepts we learned above to solve the problem. We will follow these steps to solve the problem.

Step 1: Load audio files
Step 2: Extract features from audio
Step 3: Convert the data to pass it in our deep learning model
Step 4: Run a deep learning model and get results

Below is a code of how I implemented these steps

Step 1 and  2 combined: Load audio files and extract features
def parser(row):
   # function to load files and extract features
   file_name = os.path.join(os.path.abspath(data_dir), 'Train', str(row.ID) + '.wav')

   # handle exception to check if there isn't a file which is corrupted
   try:
      # here kaiser_fast is a technique used for faster extraction
      X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
      # we extract mfcc feature from data
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
   except Exception as e:
      print("Error encountered while parsing file: ", file)
      return None, None
 
   feature = mfccs
   label = row.Class
 
   return [feature, label]

temp = train.apply(parser, axis=1)
temp.columns = ['feature', 'label']
 

Step 3: Convert the data to pass it in our deep learning model
from sklearn.preprocessing import LabelEncoder

X = np.array(temp.feature.tolist())
y = np.array(temp.label.tolist())

lb = LabelEncoder()

y = np_utils.to_categorical(lb.fit_transform(y))
Step 4: Run a deep learning model and get results
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

num_labels = y.shape[1]
filter_size = 2

# build model
model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
<!---
Vyshnavi-1107/Vyshnavi-1107 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
